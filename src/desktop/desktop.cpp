// Zorlo Desktop: Zorlo in its own window, with a tray icon, instead of a console window and a browser tab.
//
// The engine is zorlo.exe next to this program, run as `zorlo start --gui` with no window, inside a Windows job
// object: when the app quits (or crashes) the engine and the model go with it. The window is WebView2 (part of
// Windows 10/11). Before the model is loaded it shows the launcher page (launcher_html.h), fed with the
// engine's JSON lines; once http://127.0.0.1:8080/health answers it shows the Zorlo app itself.
//
// Tray: Open Zorlo, Open in browser, Switch model..., Start with Windows, Stop Zorlo and quit.
// Closing the window asks once whether to keep Zorlo running in the tray or stop it.
// Command line: --tray (start hidden, used by "Start with Windows").

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <commctrl.h>
#include <winhttp.h>
#include <wrl.h>
#include <WebView2.h>

#include "launcher_html.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr UINT WM_APP_LINE = WM_APP + 1, WM_APP_READY = WM_APP + 2, WM_APP_EXITED = WM_APP + 3, WM_APP_TRAY = WM_APP + 4,
               WM_APP_SHOW = WM_APP + 5;
constexpr UINT ID_OPEN = 1, ID_BROWSER = 2, ID_SWITCH = 3, ID_AUTOSTART = 4, ID_QUIT = 5;
const wchar_t * WND_CLASS = L"ZorloDesktopWindow";
const wchar_t * APP_URL = L"http://127.0.0.1:8080/";
const wchar_t * RUN_KEY = L"Software\\Microsoft\\Windows\\CurrentVersion\\Run";
const wchar_t * PREF_KEY = L"Software\\Zorlo";

HINSTANCE g_inst;
HWND g_wnd;
UINT g_taskbar_created;
ComPtr<ICoreWebView2Controller> g_ctrl;
ComPtr<ICoreWebView2> g_web;
bool g_page_ready = false, g_on_app = false, g_external = false, g_quitting = false, g_tip_shown = false;
std::vector<std::string> g_pending;  // engine lines that arrived before the launcher page said hello

// the engine
HANDLE g_job = nullptr, g_proc = nullptr, g_in = nullptr, g_out = nullptr;
std::thread g_reader, g_watch;
std::atomic<int> g_gen{ 0 };  // bumped on every (re)start, so old threads stop reporting

std::wstring widen(const std::string & s) {
    std::wstring w(MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), (int) w.size());
    return w;
}
std::string narrow(const std::wstring & w) {
    std::string s(WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), (int) s.size(), nullptr, nullptr);
    return s;
}
fs::path exe_dir() {
    wchar_t b[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, b, MAX_PATH * 2);
    return fs::path(std::wstring(b, n)).parent_path();
}
fs::path local_dir() {  // %LOCALAPPDATA%\Zorlo: the window's browser data and the engine log
    PWSTR p = nullptr;
    fs::path d = SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &p)) ? fs::path(p) / L"Zorlo" : exe_dir();
    CoTaskMemFree(p);
    std::error_code ec;
    fs::create_directories(d, ec);
    return d;
}
fs::path log_path() { return local_dir() / L"engine.log"; }

bool health_ok() {
    bool ok = false;
    HINTERNET s = WinHttpOpen(L"Zorlo Desktop", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s) return false;
    WinHttpSetTimeouts(s, 800, 800, 800, 800);
    if (HINTERNET c = WinHttpConnect(s, L"127.0.0.1", 8080, 0)) {
        if (HINTERNET r = WinHttpOpenRequest(c, L"GET", L"/health", nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0)) {
            DWORD code = 0, sz = sizeof code;
            if (WinHttpSendRequest(r, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(r, nullptr) &&
                WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &sz, WINHTTP_NO_HEADER_INDEX))
                ok = code == 200;
            WinHttpCloseHandle(r);
        }
        WinHttpCloseHandle(c);
    }
    WinHttpCloseHandle(s);
    return ok;
}

// ---------- page ----------
void to_page(const std::string & json_line) {
    if (!g_web || g_on_app) return;
    if (!g_page_ready) { g_pending.push_back(json_line); return; }
    g_web->PostWebMessageAsJson(widen(json_line).c_str());
}
void show_launcher() {
    g_on_app = false;
    g_page_ready = false;
    if (g_web) g_web->NavigateToString(widen(std::string(LAUNCHER_HTML_1) + LAUNCHER_HTML_2).c_str());
}
void show_app() {
    g_on_app = true;
    g_pending.clear();
    if (g_web) g_web->Navigate(APP_URL);
}
void show_window() {
    ShowWindow(g_wnd, IsIconic(g_wnd) ? SW_RESTORE : SW_SHOW);
    SetForegroundWindow(g_wnd);
}

// ---------- engine ----------
void stop_engine() {
    ++g_gen;
    if (g_job) { CloseHandle(g_job); g_job = nullptr; }  // kill-on-close: the engine and its model are gone
    if (g_in) { CloseHandle(g_in); g_in = nullptr; }
    if (g_proc) { WaitForSingleObject(g_proc, 5000); CloseHandle(g_proc); g_proc = nullptr; }
    if (g_reader.joinable()) g_reader.join();
    if (g_out) { CloseHandle(g_out); g_out = nullptr; }
    if (g_watch.joinable()) g_watch.join();
}

void start_engine(bool menu) {
    stop_engine();
    const int gen = ++g_gen;
    const fs::path exe = exe_dir() / L"zorlo.exe";
    std::wstring cmd = L"\"" + exe.wstring() + L"\" start --gui" + (menu ? L" --menu" : L"");
    SECURITY_ATTRIBUTES sa = { sizeof sa, nullptr, TRUE };
    HANDLE in_r, in_w, out_r, out_w;
    CreatePipe(&in_r, &in_w, &sa, 0);
    CreatePipe(&out_r, &out_w, &sa, 0);
    SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    HANDLE log = CreateFileW(log_path().c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, 0, nullptr);
    STARTUPINFOW si = { sizeof si };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in_r; si.hStdOutput = out_w; si.hStdError = log;
    PROCESS_INFORMATION pi = {};
    const BOOL ok = CreateProcessW(exe.c_str(), cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                                   exe_dir().c_str(), &si, &pi);
    CloseHandle(in_r); CloseHandle(out_w);
    if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
    if (!ok) {
        CloseHandle(in_w); CloseHandle(out_r);
        to_page(json{ { "type", "error" }, { "text", "Could not start zorlo.exe. It must be in the same folder as Zorlo Desktop." } }.dump());
        return;
    }
    g_job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(g_job, JobObjectExtendedLimitInformation, &li, sizeof li);
    AssignProcessToJobObject(g_job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    g_proc = pi.hProcess; g_in = in_w; g_out = out_r;

    g_reader = std::thread([gen, out = g_out, proc = g_proc] {
        std::string buf;
        char chunk[4096];
        DWORD n;
        while (ReadFile(out, chunk, sizeof chunk, &n, nullptr) && n) {
            buf.append(chunk, n);
            for (size_t p; (p = buf.find('\n')) != std::string::npos;) {
                std::string line = buf.substr(0, p);
                buf.erase(0, p + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty() && line[0] == '{' && gen == g_gen) PostMessageW(g_wnd, WM_APP_LINE, 0, (LPARAM) new std::string(line));
            }
        }
        WaitForSingleObject(proc, INFINITE);
        DWORD code = 0;
        GetExitCodeProcess(proc, &code);
        if (gen == g_gen) PostMessageW(g_wnd, WM_APP_EXITED, code, 0);
    });
}

void watch_health() {  // after "starting": wait for the server, then show the app
    if (g_watch.joinable()) g_watch.join();
    const int gen = g_gen;
    g_watch = std::thread([gen] {
        for (int i = 0; i < 1800 && gen == g_gen; ++i) {  // up to 30 minutes on a very slow disk
            if (health_ok()) { if (gen == g_gen) PostMessageW(g_wnd, WM_APP_READY, 0, 0); return; }
            Sleep(1000);
        }
    });
}

void on_engine_line(const std::string & line) {
    std::string type;
    try { type = json::parse(line).value("type", ""); } catch (...) { return; }
    if (type == "starting") watch_health();
    to_page(line);
}

void on_page_message(const std::string & msg) {
    json m;
    try { m = json::parse(msg); } catch (...) { return; }
    const std::string type = m.value("type", "");
    if (type == "hello") {
        g_page_ready = true;
        for (const auto & l : g_pending) g_web->PostWebMessageAsJson(widen(l).c_str());
        g_pending.clear();
        if (!g_proc && !g_external) start_engine(false);
    } else if (type == "choose" && g_in) {
        const std::string line = m.value("id", "") + "\n";
        DWORD w;
        WriteFile(g_in, line.data(), (DWORD) line.size(), &w, nullptr);
    } else if (type == "locate" && g_in) {  // "I already have this model": pick its folder, the engine checks it
        std::wstring folder;
        ComPtr<IFileOpenDialog> dlg;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg)))) {
            DWORD opts = 0;
            dlg->GetOptions(&opts);
            dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
            dlg->SetTitle(L"Choose the folder that has the model (manifest.json, core.gguf, experts.bin)");
            ComPtr<IShellItem> item;
            PWSTR path = nullptr;
            if (SUCCEEDED(dlg->Show(g_wnd)) && SUCCEEDED(dlg->GetResult(&item)) && SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                folder = path;
                CoTaskMemFree(path);
            }
        }
        if (!folder.empty()) {
            const std::string line = "locate " + m.value("id", "") + "\t" + narrow(folder) + "\n";
            DWORD w;
            WriteFile(g_in, line.data(), (DWORD) line.size(), &w, nullptr);
        }
    } else if (type == "retry") { show_launcher(); start_engine(false); }
    else if (type == "menu") { show_launcher(); start_engine(true); }
    else if (type == "quit") { g_quitting = true; DestroyWindow(g_wnd); }
}

// ---------- settings ----------
std::wstring pref(const wchar_t * name) {
    wchar_t v[64] = L"";
    DWORD sz = sizeof v;
    RegGetValueW(HKEY_CURRENT_USER, PREF_KEY, name, RRF_RT_REG_SZ, nullptr, v, &sz);
    return v;
}
void set_pref(const wchar_t * name, const std::wstring & v) {
    HKEY k;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, PREF_KEY, 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) == ERROR_SUCCESS) {
        RegSetValueExW(k, name, 0, REG_SZ, (const BYTE *) v.c_str(), (DWORD) ((v.size() + 1) * sizeof(wchar_t)));
        RegCloseKey(k);
    }
}
bool autostart_on() {
    DWORD sz = 0;
    return RegGetValueW(HKEY_CURRENT_USER, RUN_KEY, L"Zorlo", RRF_RT_REG_SZ, nullptr, nullptr, &sz) == ERROR_SUCCESS;
}
void set_autostart(bool on) {
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, RUN_KEY, 0, KEY_SET_VALUE, &k) != ERROR_SUCCESS) return;
    if (on) {
        wchar_t b[MAX_PATH * 2];
        GetModuleFileNameW(nullptr, b, MAX_PATH * 2);
        const std::wstring v = L"\"" + std::wstring(b) + L"\" --tray";
        RegSetValueExW(k, L"Zorlo", 0, REG_SZ, (const BYTE *) v.c_str(), (DWORD) ((v.size() + 1) * sizeof(wchar_t)));
    } else RegDeleteValueW(k, L"Zorlo");
    RegCloseKey(k);
}

// ---------- tray ----------
NOTIFYICONDATAW g_nid = { sizeof g_nid };
void tray_add() {
    g_nid.hWnd = g_wnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_ICON | NIF_TIP | NIF_MESSAGE | NIF_SHOWTIP;
    g_nid.uCallbackMessage = WM_APP_TRAY;
    g_nid.hIcon = (HICON) LoadImageW(g_inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0);
    wcscpy_s(g_nid.szTip, L"Zorlo");
    Shell_NotifyIconW(NIM_ADD, &g_nid);
    g_nid.uVersion = NOTIFYICON_VERSION_4;
    Shell_NotifyIconW(NIM_SETVERSION, &g_nid);
}
void tray_tip(const wchar_t * title, const wchar_t * text) {
    NOTIFYICONDATAW n = g_nid;
    n.uFlags = NIF_INFO;
    wcscpy_s(n.szInfoTitle, title);
    wcscpy_s(n.szInfo, text);
    n.dwInfoFlags = NIIF_USER | NIIF_LARGE_ICON;
    n.hBalloonIcon = (HICON) LoadImageW(g_inst, MAKEINTRESOURCEW(1), IMAGE_ICON, 32, 32, 0);
    Shell_NotifyIconW(NIM_MODIFY, &n);
}
void tray_menu() {
    HMENU m = CreatePopupMenu();
    AppendMenuW(m, MF_STRING, ID_OPEN, L"Open Zorlo");
    AppendMenuW(m, MF_STRING | (g_on_app ? 0 : MF_GRAYED), ID_BROWSER, L"Open in browser");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING | (g_external ? MF_GRAYED : 0), ID_SWITCH, L"Switch model\x2026");
    AppendMenuW(m, MF_STRING | (autostart_on() ? MF_CHECKED : 0), ID_AUTOSTART, L"Start with Windows (loads the model at sign-in)");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, ID_QUIT, g_external ? L"Close Zorlo Desktop" : L"Stop Zorlo and quit");
    SetMenuDefaultItem(m, ID_OPEN, FALSE);
    POINT p;
    GetCursorPos(&p);
    SetForegroundWindow(g_wnd);
    const UINT id = TrackPopupMenu(m, TPM_RETURNCMD | TPM_RIGHTBUTTON | TPM_BOTTOMALIGN, p.x, p.y, 0, g_wnd, nullptr);
    DestroyMenu(m);
    switch (id) {
        case ID_OPEN: show_window(); break;
        case ID_BROWSER: ShellExecuteW(nullptr, L"open", APP_URL, nullptr, nullptr, SW_SHOWNORMAL); break;
        case ID_SWITCH: show_window(); show_launcher(); start_engine(true); break;
        case ID_AUTOSTART: set_autostart(!autostart_on()); break;
        case ID_QUIT: g_quitting = true; DestroyWindow(g_wnd); break;
    }
}

// closing the window: keep running in the tray, or stop (asked once, unless "don't ask again")
bool keep_running_on_close() {
    const std::wstring saved = pref(L"OnClose");
    if (saved == L"tray") return true;
    if (saved == L"quit") return false;
    TASKDIALOG_BUTTON buttons[] = { { 100, L"Keep running in the background\nThe model stays loaded; open Zorlo again from the icon next to the clock." },
                                    { 101, L"Stop Zorlo\nUnloads the model and frees its memory." } };
    TASKDIALOGCONFIG c = { sizeof c };
    c.hwndParent = g_wnd;
    c.hInstance = g_inst;
    c.dwFlags = TDF_USE_COMMAND_LINKS | TDF_ALLOW_DIALOG_CANCELLATION | TDF_POSITION_RELATIVE_TO_WINDOW;
    c.pszWindowTitle = L"Zorlo";
    c.pszMainIcon = MAKEINTRESOURCEW(1);
    c.pszMainInstruction = L"Keep Zorlo running?";
    c.pButtons = buttons;
    c.cButtons = 2;
    c.nDefaultButton = 100;
    c.pszVerificationText = L"Don't ask again";
    int choice = 0;
    BOOL remember = FALSE;
    if (FAILED(TaskDialogIndirect(&c, &choice, nullptr, &remember)) || choice == IDCANCEL) return true;
    if (remember) set_pref(L"OnClose", choice == 100 ? L"tray" : L"quit");
    return choice == 100;
}

// ---------- WebView2 ----------
void resize() {
    if (!g_ctrl) return;
    RECT r;
    GetClientRect(g_wnd, &r);
    g_ctrl->put_Bounds(r);
}

void no_webview() {  // WebView2 runtime missing (rare on Windows 10/11): fall back to the browser
    MessageBoxW(g_wnd, L"Zorlo Desktop needs Microsoft Edge WebView2, which is part of Windows 10 and 11 but is missing or "
                       L"broken on this PC.\n\nZorlo will open in your browser instead. To get the app window, install the "
                       L"\"WebView2 Runtime\" from Microsoft's website.",
                L"Zorlo", MB_ICONINFORMATION);
    const fs::path exe = exe_dir() / L"zorlo.exe";
    ShellExecuteW(nullptr, L"open", exe.c_str(), L"start", exe_dir().c_str(), SW_SHOWNORMAL);
    g_quitting = true;
    DestroyWindow(g_wnd);
}

bool is_local(const std::wstring & uri) {
    return uri.rfind(L"http://127.0.0.1:8080", 0) == 0 || uri.rfind(L"about:", 0) == 0 || uri.rfind(L"data:", 0) == 0;
}

void create_webview() {
    const std::wstring udf = (local_dir() / L"WebView2").wstring();
    CreateCoreWebView2EnvironmentWithOptions(nullptr, udf.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>([](HRESULT hr, ICoreWebView2Environment * env) -> HRESULT {
            if (FAILED(hr) || !env) { no_webview(); return S_OK; }
            return env->CreateCoreWebView2Controller(g_wnd,
                Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>([](HRESULT hr2, ICoreWebView2Controller * c) -> HRESULT {
                    if (FAILED(hr2) || !c) { no_webview(); return S_OK; }
                    g_ctrl = c;
                    g_ctrl->get_CoreWebView2(&g_web);
                    COREWEBVIEW2_COLOR bg = { 255, 0xF5, 0xF3, 0xEE };
                    ComPtr<ICoreWebView2Controller2> c2;
                    if (SUCCEEDED(g_ctrl.As(&c2))) c2->put_DefaultBackgroundColor(bg);
                    resize();
                    ComPtr<ICoreWebView2Settings> s;
                    g_web->get_Settings(&s);
                    s->put_AreDevToolsEnabled(FALSE);
                    s->put_IsStatusBarEnabled(FALSE);
                    s->put_IsZoomControlEnabled(TRUE);
                    EventRegistrationToken t;
                    // only the launcher page may talk to the app
                    g_web->add_WebMessageReceived(Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                        [](ICoreWebView2 *, ICoreWebView2WebMessageReceivedEventArgs * a) -> HRESULT {
                            LPWSTR src = nullptr, s2 = nullptr;
                            a->get_Source(&src);
                            const bool launcher = src && (wcsncmp(src, L"about:", 6) == 0 || wcsncmp(src, L"data:", 5) == 0);
                            CoTaskMemFree(src);
                            if (launcher && SUCCEEDED(a->get_WebMessageAsJson(&s2))) { on_page_message(narrow(s2)); CoTaskMemFree(s2); }
                            return S_OK;
                        }).Get(), &t);
                    // other websites open in the normal browser, never inside Zorlo
                    g_web->add_NewWindowRequested(Callback<ICoreWebView2NewWindowRequestedEventHandler>(
                        [](ICoreWebView2 *, ICoreWebView2NewWindowRequestedEventArgs * a) -> HRESULT {
                            LPWSTR uri = nullptr;
                            a->get_Uri(&uri);
                            if (uri) { ShellExecuteW(nullptr, L"open", uri, nullptr, nullptr, SW_SHOWNORMAL); CoTaskMemFree(uri); }
                            a->put_Handled(TRUE);
                            return S_OK;
                        }).Get(), &t);
                    g_web->add_NavigationStarting(Callback<ICoreWebView2NavigationStartingEventHandler>(
                        [](ICoreWebView2 *, ICoreWebView2NavigationStartingEventArgs * a) -> HRESULT {
                            LPWSTR uri = nullptr;
                            a->get_Uri(&uri);
                            if (uri && !is_local(uri)) { ShellExecuteW(nullptr, L"open", uri, nullptr, nullptr, SW_SHOWNORMAL); a->put_Cancel(TRUE); }
                            CoTaskMemFree(uri);
                            return S_OK;
                        }).Get(), &t);
                    if (g_external) show_app(); else show_launcher();
                    return S_OK;
                }).Get());
        }).Get());
}

// Start menu → Zorlo, like an installed app. Kept pointing at this copy: if the folder moved, the entry is updated.
void ensure_start_menu_entry() {
    PWSTR progs = nullptr;
    if (FAILED(SHGetKnownFolderPath(FOLDERID_Programs, 0, nullptr, &progs))) return;
    const fs::path lnk = fs::path(progs) / L"Zorlo.lnk";
    CoTaskMemFree(progs);
    wchar_t self[MAX_PATH * 2];
    GetModuleFileNameW(nullptr, self, MAX_PATH * 2);
    ComPtr<IShellLinkW> link;
    if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&link)))) return;
    ComPtr<IPersistFile> file;
    if (FAILED(link.As(&file))) return;
    std::error_code ec;
    if (fs::exists(lnk, ec) && SUCCEEDED(file->Load(lnk.c_str(), STGM_READ))) {
        wchar_t target[MAX_PATH * 2] = L"";
        link->GetPath(target, MAX_PATH * 2, nullptr, 0);
        if (_wcsicmp(target, self) == 0) return;  // already this copy
    }
    link->SetPath(self);
    link->SetWorkingDirectory(exe_dir().c_str());
    link->SetDescription(L"Zorlo: AI models on your own PC");
    link->SetIconLocation(self, 0);
    file->Save(lnk.c_str(), TRUE);
}

LRESULT CALLBACK wnd_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == g_taskbar_created && g_taskbar_created) { tray_add(); return 0; }  // Explorer restarted
    switch (msg) {
        case WM_SIZE: resize(); return 0;
        case WM_GETMINMAXINFO: { auto * mm = (MINMAXINFO *) lp; mm->ptMinTrackSize = { 520, 460 }; return 0; }
        case WM_APP_LINE: { std::unique_ptr<std::string> l((std::string *) lp); on_engine_line(*l); return 0; }
        case WM_APP_READY: show_app(); return 0;
        case WM_APP_EXITED:
            if (!g_quitting) {
                if (g_on_app) { show_launcher(); show_window(); }
                to_page(json{ { "type", "exited" }, { "code", (int) wp }, { "log", narrow(log_path().wstring()) } }.dump());
            }
            return 0;
        case WM_APP_SHOW: show_window(); return 0;
        case WM_APP_TRAY:
            switch (LOWORD(lp)) {
                case WM_LBUTTONUP: case NIN_SELECT: case NIN_KEYSELECT: show_window(); break;
                case WM_CONTEXTMENU: case WM_RBUTTONUP: tray_menu(); break;
            }
            return 0;
        case WM_CLOSE:
            if (g_quitting || !keep_running_on_close()) { g_quitting = true; DestroyWindow(h); return 0; }
            ShowWindow(h, SW_HIDE);
            if (!g_tip_shown) { tray_tip(L"Zorlo is still running", L"The model stays loaded. Click the Zorlo icon to open it, or right-click to stop it."); g_tip_shown = true; }
            return 0;
        case WM_DESTROY:
            g_quitting = true;
            Shell_NotifyIconW(NIM_DELETE, &g_nid);
            if (g_ctrl) g_ctrl->Close();
            stop_engine();
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR cmdline, int show) {
    g_inst = inst;
    // one Zorlo Desktop at a time: a second start just brings the first one forward
    HANDLE single = CreateMutexW(nullptr, TRUE, L"Local\\ZorloDesktop");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        if (HWND other = FindWindowW(WND_CLASS, nullptr)) PostMessageW(other, WM_APP_SHOW, 0, 0);
        return 0;
    }
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    INITCOMMONCONTROLSEX icc = { sizeof icc, ICC_STANDARD_CLASSES };
    InitCommonControlsEx(&icc);

    WNDCLASSEXW wc = { sizeof wc };
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = inst;
    wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));
    wc.hIconSm = (HICON) LoadImageW(inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), 0);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = CreateSolidBrush(RGB(0xF5, 0xF3, 0xEE));
    wc.lpszClassName = WND_CLASS;
    RegisterClassExW(&wc);

    // about 1280 x 820 at 100% scaling, never bigger than the work area
    RECT wa;
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0);
    const UINT dpi = GetDpiForSystem();
    const int w = std::min<int>(MulDiv(1280, dpi, 96), wa.right - wa.left - 40), hgt = std::min<int>(MulDiv(820, dpi, 96), wa.bottom - wa.top - 40);
    g_wnd = CreateWindowExW(0, WND_CLASS, L"Zorlo", WS_OVERLAPPEDWINDOW, wa.left + (wa.right - wa.left - w) / 2,
                            wa.top + (wa.bottom - wa.top - hgt) / 2, w, hgt, nullptr, nullptr, inst, nullptr);
    g_taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    tray_add();
    ensure_start_menu_entry();

    g_external = health_ok();  // a Zorlo started some other way is already running: just show it
    const bool tray_only = cmdline && wcsstr(cmdline, L"--tray");
    if (!tray_only) { ShowWindow(g_wnd, show); UpdateWindow(g_wnd); }
    create_webview();

    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0)) { TranslateMessage(&m); DispatchMessageW(&m); }
    CoUninitialize();
    if (single) CloseHandle(single);
    return 0;
}
