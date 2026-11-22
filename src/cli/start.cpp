// zorlo start: the double-click experience ("Start Zorlo.bat" runs it) and the engine side of Zorlo Desktop.
//
//   1. reads the model catalog: models.json from the Zorlo catalog repo on Hugging Face, so new models show up
//      without a new Zorlo download; the list built in below is used when that is unreachable
//   2. moves a model downloaded earlier into another folder into models\<id> (never download twice)
//   3. shows the models already downloaded and the ones that can be downloaded; models that need a newer
//      Zorlo are listed with a note instead of a number
//   4. downloads the chosen model (or updates it when its files changed on the Hub) into <zorlo>\models\<id>
//   5. serves it and opens the chat page in the browser once the server answers
//
// --gui (used by Zorlo Desktop): JSON lines on stdout, the choice on stdin, no browser. --menu: always ask.
// Overrides for testing: ZORLO_CATALOG (https URL or local file), ZORLO_MODELS_DIR, ZORLO_NO_BROWSER=1.

#include "cli/pull.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <string>

int zorlo_serve_main(int argc, char ** argv);  // src/cli/main.cpp

#ifndef _WIN32
int zorlo_start_main(int, char **) {
    fprintf(stderr, "zorlo start: only on Windows for now; use zorlo pull and zorlo serve\n");
    return 1;
}
int zorlo_path_main(int, char **) {
    fprintf(stderr, "zorlo path: only on Windows for now\n");
    return 1;
}
#else

#include "httplib.h"  // before windows.h (winsock order)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

static const char * ZORLO_VERSION = "0.1";
static const char * CATALOG_URL = "https://huggingface.co/datasets/Haidershahi1/zorlo-catalog/resolve/main/models.json";

// used when the online catalog can't be read (offline, or not published yet)
static const char * BUILTIN_CATALOG = R"({
  "format": "zorlo-catalog-1",
  "models": [
    { "id": "qwen3-30b-a3b", "name": "Qwen3 30B", "repo": "Haidershahi1/qwen3-30b-a3b-zorlo",
      "download_gb": 18.6, "min_ram_gb": 8, "recommended": true, "min_zorlo": "0.1",
      "about": "General chat, writing and coding. Works on 8 GB PCs." }
  ]
})";

struct Model {
    std::string id, name, repo, about, min_zorlo = "0";
    double gb = 0, ram = 0;
    bool recommended = false;
};

static int version_cmp(const std::string & a, const std::string & b) {
    std::istringstream sa(a), sb(b);
    std::string pa, pb;
    for (;;) {
        const bool ha = (bool) std::getline(sa, pa, '.'), hb = (bool) std::getline(sb, pb, '.');
        if (!ha && !hb) return 0;
        const long va = ha ? std::atol(pa.c_str()) : 0, vb = hb ? std::atol(pb.c_str()) : 0;
        if (va != vb) return va < vb ? -1 : 1;
    }
}

static bool safe_id(const std::string & s) {
    return !s.empty() && s.size() < 80 && s[0] != '.' &&
           std::all_of(s.begin(), s.end(), [](char c) { return isalnum((unsigned char) c) || c == '-' || c == '_' || c == '.'; });
}

static std::vector<Model> parse_catalog(std::string text) {
    if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);  // UTF-8 BOM (Notepad, PowerShell 5.1)
    const json j = json::parse(text);
    if (j.value("format", "") != "zorlo-catalog-1") throw std::runtime_error("unknown catalog format");
    std::vector<Model> out;
    for (const auto & e : j.at("models")) {
        Model m;
        m.id = e.value("id", ""); m.repo = e.value("repo", "");
        if (!safe_id(m.id) || m.repo.find('/') == std::string::npos) continue;
        m.name = e.value("name", m.id); m.about = e.value("about", "");
        m.gb = e.value("download_gb", 0.0); m.ram = e.value("min_ram_gb", 0.0);
        m.recommended = e.value("recommended", false); m.min_zorlo = e.value("min_zorlo", "0");
        out.push_back(m);
    }
    if (out.empty()) throw std::runtime_error("catalog lists no models");
    return out;
}

// the catalog also says which Zorlo is the newest: {"zorlo": {"latest": "0.2", "url": "...", "notes": "..."}}
static json g_latest;
static std::vector<Model> load_catalog() {
    std::string text, err, src = CATALOG_URL;
    if (const char * o = getenv("ZORLO_CATALOG")) src = o;
    bool ok;
    if (src.rfind("https://", 0) == 0) ok = zorlo_fetch_text(src, text, err);
    else { std::ifstream in(fs::u8path(src)); std::stringstream ss; ss << in.rdbuf(); text = ss.str(); ok = (bool) in; }
    if (ok) {
        try {
            auto models = parse_catalog(text);
            if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
            const json j = json::parse(text);
            if (j.contains("zorlo") && j["zorlo"].is_object()) g_latest = j["zorlo"];
            return models;
        } catch (const std::exception & e) { err = e.what(); }
    }
    return parse_catalog(BUILTIN_CATALOG);  // offline or no online catalog yet: the models this version knows
}
static void say(const std::string & human, const json & j);  // below
static void tell_if_newer() {
    if (!g_latest.is_object()) return;
    const std::string latest = g_latest.value("latest", "");
    if (latest.empty() || version_cmp(ZORLO_VERSION, latest) >= 0) return;
    const std::string url = g_latest.value("url", ""), notes = g_latest.value("notes", "");
    say("  Zorlo " + latest + " is available (you have " + ZORLO_VERSION + ")" + (notes.empty() ? "" : ": " + notes) + ".\n  " +
            (url.empty() ? "Download it from the Zorlo website." : "Download it from " + url) + "\n\n",
        { { "type", "note" }, { "text", "Zorlo " + latest + " is available (you have " + ZORLO_VERSION + "). " +
                                            (url.empty() ? "Download it from the Zorlo website." : "Download it from " + url) } });
}

static fs::path exe_dir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof buf / sizeof buf[0]));
    return fs::path(std::wstring(buf, n)).parent_path();
}

static bool has_part_files(const fs::path & d) {
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(d, ec))
        if (e.path().extension() == ".part") return true;
    return false;
}

static void open_browser_when_ready(int port) {
    std::thread([port] {
        httplib::Client cli("127.0.0.1", port);
        cli.set_connection_timeout(1, 0);
        cli.set_read_timeout(2, 0);
        for (int i = 0; i < 900; ++i) {  // up to 15 minutes (a slow disk loading the model)
            if (auto res = cli.Get("/health"); res && res->status == 200) {
                const std::wstring url = L"http://127.0.0.1:" + std::to_wstring(port);
                ShellExecuteW(nullptr, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
                return;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }).detach();
}

// --gui: for Zorlo Desktop. Everything the user sees goes to stdout as one JSON object per line
// (models, notes, progress, starting, error); the chosen model id (or "quit") comes back on stdin.
static bool g_gui = false;
static void emit(const json & j) { printf("%s\n", j.dump(-1, ' ', false, json::error_handler_t::replace).c_str()); fflush(stdout); }
static void say(const std::string & human, const json & j) { if (g_gui) emit(j); else { printf("%s", human.c_str()); fflush(stdout); } }

// ---------- never download a model twice ----------
// A model already on this PC (downloaded with `zorlo pull` into another folder, or made with `zorlo pack`) is used
// where it is: models\links.json remembers its folder. Nothing is moved or copied (an 18 GB copy across drives).
static std::string norm(const std::string & s) {
    std::string o;
    for (unsigned char c : s) if (isalnum(c)) o += (char) tolower(c);
    return o;
}
static json read_json(const fs::path & p) {
    std::ifstream in(p);
    if (!in) return nullptr;
    try { return json::parse(in); } catch (...) { return nullptr; }
}
static bool is_pack(const fs::path & d) {
    const json m = read_json(d / "manifest.json");
    if (!m.is_object()) return false;
    const std::string f = m.value("format", "");
    return f == "zorlo-pack-1" || f == "zoro-pack-1";
}
// which catalog model a pack folder holds: its download record, else what it was made from (or the folder's name)
static std::string pack_model(const fs::path & d, const std::vector<Model> & catalog) {
    const json pull = read_json(d / ".zorlo-pull.json");
    if (pull.is_object()) {
        const std::string repo = pull.value("repo", "");
        for (const auto & m : catalog) if (repo == m.repo || repo.rfind(m.repo + "@", 0) == 0) return m.id;
    }
    const json man = read_json(d / "manifest.json");
    const std::string src = norm(man.is_object() ? man.value("source", "") : "") + " " + norm(d.filename().u8string());
    for (const auto & m : catalog) if (!norm(m.id).empty() && src.find(norm(m.id)) != std::string::npos) return m.id;
    return "";
}
static std::map<std::string, std::string> load_links(const fs::path & models_dir) {
    std::map<std::string, std::string> l;
    const json j = read_json(models_dir / "links.json");
    if (j.is_object()) for (const auto & [k, v] : j.items()) if (v.is_string()) l[k] = v.get<std::string>();
    return l;
}
static void save_links(const fs::path & models_dir, const std::map<std::string, std::string> & l) {
    std::ofstream(models_dir / "links.json") << json(l).dump(2);
}
// the folder that holds model `id`: models\<id>, else a remembered folder; models\<id> is also where a download goes
static bool pack_ready(const fs::path & d) { return is_pack(d) && !has_part_files(d); }
// the folder that holds model `id`: a complete copy in models\<id>, else a complete copy elsewhere that Zorlo found
// or was shown, else models\<id> (where a download goes, or continues). An unfinished download never hides a
// complete copy.
static fs::path model_dir(const fs::path & models_dir, const std::string & id, const std::map<std::string, std::string> & links) {
    const fs::path own = models_dir / fs::u8path(id);
    if (pack_ready(own)) return own;
    const auto f = links.find(id);
    if (f != links.end() && pack_ready(fs::u8path(f->second))) return fs::u8path(f->second);
    return own;
}
// look for packs of the models not on hand yet: next to Zorlo, in Downloads / Documents / Desktop, and two folders
// deep on every local drive (E:\models\qwen3.zorlo). Only folder names are read, so this takes a moment at most.
static void find_existing(const std::vector<Model> & catalog, const fs::path & models_dir, std::map<std::string, std::string> & links) {
    std::error_code ec;
    std::vector<std::string> want;
    for (const auto & m : catalog) if (!pack_ready(model_dir(models_dir, m.id, links))) want.push_back(m.id);
    if (want.empty()) return;
    std::vector<fs::path> roots = { exe_dir(), fs::current_path(ec) };
    if (const char * up = getenv("USERPROFILE"))
        for (const char * sub : { "Downloads", "Documents", "Desktop" }) roots.push_back(fs::u8path(up) / sub);
    const DWORD drives = GetLogicalDrives();
    for (int i = 0; i < 26; ++i) {
        if (!(drives & (1u << i))) continue;
        const std::wstring root = std::wstring(1, (wchar_t) (L'A' + i)) + L":\\";
        if (GetDriveTypeW(root.c_str()) == DRIVE_FIXED) roots.push_back(fs::path(root));
    }
    static const char * skip[] = { "windows", "program files", "program files (x86)", "programdata", "appdata", "$recycle.bin",
                                   "system volume information", "recovery", "msocache", "perflogs", "node_modules", ".git", "windowsapps" };
    auto skipped = [](const fs::path & p) {
        std::string n = p.filename().u8string();
        std::transform(n.begin(), n.end(), n.begin(), ::tolower);
        return n.empty() || n[0] == '.' || n[0] == '$' || std::any_of(std::begin(skip), std::end(skip), [&](const char * s) { return n == s; });
    };
    std::map<std::string, std::vector<fs::path>> found;
    std::function<void(const fs::path &, int)> walk = [&](const fs::path & d, int depth) {
        for (fs::directory_iterator it(d, fs::directory_options::skip_permission_denied, ec), end; it != end; it.increment(ec)) {
            if (ec) break;
            if (!it->is_directory(ec) || skipped(it->path())) continue;
            if (is_pack(it->path())) {
                const std::string id = pack_model(it->path(), catalog);
                if (!id.empty() && !has_part_files(it->path())) found[id].push_back(it->path());
            } else if (depth < 2) walk(it->path(), depth + 1);
        }
    };
    for (const auto & r : roots) if (fs::is_directory(r, ec)) walk(r, 1);
    bool changed = false;
    for (const auto & id : want) {
        const auto f = found.find(id);
        if (f == found.end()) continue;
        fs::path best = f->second[0];  // one with a download record can be kept up to date
        for (const auto & p : f->second) if (fs::exists(p / ".zorlo-pull.json", ec)) { best = p; break; }
        links[id] = best.u8string();
        changed = true;
        const auto m = std::find_if(catalog.begin(), catalog.end(), [&](const Model & x) { return x.id == id; });
        say("  Found " + m->name + " already on this PC in " + best.u8string() + "; using it from there.\n",
            { { "type", "note" }, { "text", "Found " + m->name + " already on this PC (" + best.u8string() + "). Zorlo uses it from there; nothing is downloaded." } });
    }
    if (changed) save_links(models_dir, links);
}

// ---------- `zorlo` from any folder ----------
// The Zorlo folder goes on the user's PATH (HKCU\Environment, no admin rights), so `zorlo start` works in any new
// terminal. HKCU\Software\Zorlo\PathDir remembers which entry is ours: when the folder moves, the old one is replaced.
// `zorlo path off` takes it out again (and stops it coming back); `zorlo path on` puts it back.
static std::wstring reg_str(HKEY root, const wchar_t * key, const wchar_t * name, DWORD * type = nullptr) {
    DWORD sz = 0, t = 0;
    if (RegGetValueW(root, key, name, RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND, &t, nullptr, &sz) != ERROR_SUCCESS) return L"";
    std::wstring v(sz / sizeof(wchar_t), L'\0');
    if (RegGetValueW(root, key, name, RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND, &t, v.data(), &sz) != ERROR_SUCCESS) return L"";
    v.resize(wcslen(v.c_str()));
    if (type) *type = t;
    return v;
}
static void set_reg_str(HKEY root, const wchar_t * key, const wchar_t * name, const std::wstring & v, DWORD type = REG_SZ) {
    HKEY k;
    if (RegCreateKeyExW(root, key, 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) != ERROR_SUCCESS) return;
    RegSetValueExW(k, name, 0, type, (const BYTE *) v.c_str(), (DWORD) ((v.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(k);
}
static std::wstring lower(std::wstring s) { std::transform(s.begin(), s.end(), s.begin(), ::towlower); return s; }
static std::wstring trim_slash(std::wstring s) { while (!s.empty() && (s.back() == L'\\' || s.back() == L'/')) s.pop_back(); return s; }

// true when the PATH changed
static bool path_register(bool on, bool quiet) {
    const wchar_t * ENV = L"Environment", * PREF = L"Software\\Zorlo";
    if (on && !quiet && reg_str(HKEY_CURRENT_USER, PREF, L"PathOff") == L"1") return false;  // the user turned it off
    const std::wstring dir = trim_slash(exe_dir().wstring()), old = trim_slash(reg_str(HKEY_CURRENT_USER, PREF, L"PathDir"));
    DWORD type = REG_EXPAND_SZ;
    const std::wstring cur = reg_str(HKEY_CURRENT_USER, ENV, L"Path", &type);
    std::vector<std::wstring> parts;
    bool have = false, changed = false;
    for (size_t a = 0; a <= cur.size();) {
        size_t b = cur.find(L';', a);
        if (b == std::wstring::npos) b = cur.size();
        std::wstring p = cur.substr(a, b - a);
        a = b + 1;
        if (p.empty()) continue;
        const std::wstring lp = lower(trim_slash(p));
        if (!old.empty() && lp == lower(old) && lp != lower(dir)) { changed = true; continue; }  // our old folder
        if (lp == lower(dir)) { if (!on) { changed = true; continue; } have = true; }
        parts.push_back(p);
    }
    if (on && !have) {
        // already reachable through the system PATH? then nothing to add
        const std::wstring sys = lower(reg_str(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment", L"Path"));
        if (sys.find(lower(dir)) == std::wstring::npos) { parts.push_back(dir); changed = true; }
    }
    if (changed) {
        std::wstring v;
        for (const auto & p : parts) v += (v.empty() ? L"" : L";") + p;
        set_reg_str(HKEY_CURRENT_USER, ENV, L"Path", v, type == REG_SZ ? REG_SZ : REG_EXPAND_SZ);
        DWORD_PTR r;
        SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, (LPARAM) L"Environment", SMTO_ABORTIFHUNG, 2000, &r);  // new terminals see it
    }
    set_reg_str(HKEY_CURRENT_USER, PREF, L"PathDir", on ? dir : L"");
    set_reg_str(HKEY_CURRENT_USER, PREF, L"PathOff", on ? L"0" : L"1");
    return changed;
}

int zorlo_path_main(int argc, char ** argv) {
    const std::string what = argc > 1 ? argv[1] : "";
    if (what != "on" && what != "off") {
        printf("usage: zorlo path on    let `zorlo` work in any terminal, from any folder\n"
               "       zorlo path off   take this Zorlo folder out of your PATH\n");
        return what.empty() ? 0 : 1;
    }
    path_register(what == "on", true);
    printf(what == "on" ? "  Done. In a new terminal window, `zorlo start` now works from any folder.\n"
                        : "  Done. `zorlo` is no longer on your PATH; run it as .\\zorlo from its folder.\n");
    return 0;
}

int zorlo_start_main(int argc, char ** argv) {
    bool force_menu = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--gui") g_gui = true;
        else if (s == "--menu") force_menu = true;
    }
    SetConsoleOutputCP(CP_UTF8);
    const fs::path models_dir = getenv("ZORLO_MODELS_DIR") ? fs::u8path(getenv("ZORLO_MODELS_DIR")) : exe_dir() / "models";
    std::error_code ec;
    fs::create_directories(models_dir, ec);

    if (!g_gui) {
        printf("\n  Zorlo %s - AI models on your own PC\n  ------------------------------------\n\n", ZORLO_VERSION);
        // already running (another window, the tray, Zorlo Desktop): open that one instead of loading the model twice
        httplib::Client probe("127.0.0.1", 8080);
        probe.set_connection_timeout(1, 0);
        probe.set_read_timeout(2, 0);
        if (auto res = probe.Get("/health"); res && res->status == 200) {
            printf("  Zorlo is already running: http://127.0.0.1:8080\n");
            if (!getenv("ZORLO_NO_BROWSER")) {
                printf("  Opening it in your browser.\n");
                ShellExecuteW(nullptr, L"open", L"http://127.0.0.1:8080", nullptr, nullptr, SW_SHOWNORMAL);
            }
            printf("\n  To stop it: close its window, use the Zorlo icon next to the clock, or run:\n      Stop-Process -Name zorlo\n\n");
            return 0;
        }
        if (!getenv("ZORLO_NO_PATH") && path_register(true, false))
            printf("  From now on you can start Zorlo from any folder: open a new terminal and type  zorlo start\n"
                   "  (undo with: zorlo path off)\n\n");
        printf("  Checking the list of models...\n");
    } else if (!getenv("ZORLO_NO_PATH")) path_register(true, false);
    const std::vector<Model> catalog = load_catalog();
    tell_if_newer();
    std::map<std::string, std::string> links = load_links(models_dir);
    find_existing(catalog, models_dir, links);

    // what's on disk: models\<id> or a remembered folder, ready unless a download is unfinished
    struct Row { Model m; fs::path dir; bool installed = false, ready = false; };
    std::vector<Row> rows;
    std::vector<Model> too_new;
    for (const auto & m : catalog) {
        Row r{ m, model_dir(models_dir, m.id, links) };
        r.installed = fs::exists(r.dir / "manifest.json", ec) || fs::exists(r.dir / ".zorlo-pull.json", ec);
        r.ready = fs::exists(r.dir / "manifest.json", ec) && !has_part_files(r.dir);
        if (!r.installed && version_cmp(ZORLO_VERSION, m.min_zorlo) < 0) { too_new.push_back(m); continue; }
        rows.push_back(r);
    }
    for (const auto & e : fs::directory_iterator(models_dir, ec)) {  // packs placed by hand, not in the catalog
        if (!e.is_directory() || !fs::exists(e.path() / "manifest.json")) continue;
        const std::string id = e.path().filename().u8string();
        if (std::any_of(rows.begin(), rows.end(), [&](const Row & r) { return r.m.id == id; })) continue;
        Row r; r.m.id = id; r.m.name = id; r.dir = e.path(); r.installed = true; r.ready = !has_part_files(e.path());
        rows.push_back(r);
    }
    std::stable_partition(rows.begin(), rows.end(), [](const Row & r) { return r.installed; });
    if (rows.empty()) {
        say("\n  No models are available for this version of Zorlo.\n", { { "type", "error" }, { "text", "No models are available for this version of Zorlo." } });
        return 1;
    }

    std::string last;
    { std::ifstream in(models_dir / "last.txt"); std::getline(in, last); }
    size_t def = 0;
    bool last_ready = false;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].installed && rows[i].m.id == last) { def = i; last_ready = rows[i].ready; break; }
        if (!rows[def].installed && !rows[def].m.recommended && rows[i].m.recommended) def = i;
    }

    size_t pick = def;
    const bool only_one = rows.size() == 1 && rows[0].ready;
    if (g_gui) {
        json items = json::array(), newer = json::array();
        for (size_t i = 0; i < rows.size(); ++i) {
            const Row & r = rows[i];
            items.push_back({ { "id", r.m.id }, { "name", r.m.name }, { "about", r.m.about }, { "gb", r.m.gb }, { "ram", r.m.ram },
                              { "installed", r.installed }, { "ready", r.ready }, { "recommended", r.m.recommended }, { "default", i == def } });
        }
        for (const auto & m : too_new) newer.push_back(m.name);
        emit({ { "type", "models" }, { "items", items }, { "too_new", newer }, { "version", ZORLO_VERSION } });
        if (!force_menu && (only_one || last_ready)) emit({ { "type", "auto" }, { "id", rows[def].m.id } });
        else for (;;) {
            std::string line;
            if (!std::getline(std::cin, line) || line == "quit") return 0;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // "locate <id>\t<folder>": the user points at a copy they already have
            if (line.rfind("locate ", 0) == 0 && line.find('\t') != std::string::npos) {
                const std::string id = line.substr(7, line.find('\t') - 7), folder = line.substr(line.find('\t') + 1);
                const auto it = std::find_if(rows.begin(), rows.end(), [&](const Row & r) { return r.m.id == id; });
                const fs::path p = fs::u8path(folder);
                std::string why;
                if (it == rows.end()) why = "unknown model";
                else if (!is_pack(p)) why = "that folder has no Zorlo model in it (it should contain manifest.json, core.gguf and experts.bin)";
                else if (has_part_files(p)) why = "the download in that folder is not finished";
                else if (const std::string got = pack_model(p, catalog); !got.empty() && got != id) why = "that folder holds a different model (" + got + ")";
                if (!why.empty()) { emit({ { "type", "located" }, { "ok", false }, { "id", id }, { "text", why } }); continue; }
                links[id] = p.u8string();
                save_links(models_dir, links);
                it->dir = p; it->installed = it->ready = true;
                pick = (size_t) (it - rows.begin());
                emit({ { "type", "located" }, { "ok", true }, { "id", id }, { "text", "Using the copy in " + folder } });
                break;
            }
            line.erase(std::remove_if(line.begin(), line.end(), [](char c) { return isspace((unsigned char) c); }), line.end());
            for (size_t i = 0; i < rows.size(); ++i) if (rows[i].m.id == line) pick = i;
            break;
        }
    } else if (!only_one) {
        const size_t n_inst = std::count_if(rows.begin(), rows.end(), [](const Row & r) { return r.installed; });
        printf("\n");
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i == 0 && n_inst) printf("  Your models\n");
            if (i == n_inst) printf(n_inst ? "\n  Download another model\n" : "  Choose a model to download\n");
            const Row & r = rows[i];
            if (r.installed) printf("    %zu  %-28s %s\n", i + 1, r.m.name.c_str(), r.ready ? "ready" : "download not finished");
            else {
                printf("    %zu  %-28s %.1f GB download", i + 1, r.m.name.c_str(), r.m.gb);
                if (r.m.ram > 0) printf(", needs %.0f GB memory", r.m.ram);
                printf("%s\n", r.m.recommended ? "  (recommended)" : "");
            }
            if (!r.m.about.empty()) printf("       %s\n", r.m.about.c_str());
        }
        if (!too_new.empty()) {
            printf("\n  Also available with a newer Zorlo (download the latest version from the website):\n");
            for (const auto & m : too_new) printf("       %s\n", m.name.c_str());
        }
        for (;;) {
            printf("\n  Type a number and press Enter (just Enter = %zu, q = quit): ", def + 1);
            fflush(stdout);
            std::string line;
            if (!std::getline(std::cin, line)) break;  // no keyboard (script): take the default
            line.erase(std::remove_if(line.begin(), line.end(), [](char c) { return isspace((unsigned char) c); }), line.end());
            if (line.empty()) break;
            if (line == "q" || line == "Q") return 0;
            char * end = nullptr;
            const long v = strtol(line.c_str(), &end, 10);
            if (*end == '\0' && v >= 1 && (size_t) v <= rows.size()) { pick = (size_t) v - 1; break; }
            printf("  Please type a number from 1 to %zu.", rows.size());
        }
    }

    const Row & r = rows[pick];
    const fs::path dir = r.dir.empty() ? models_dir / fs::u8path(r.m.id) : r.dir;
    // a pack made on this PC (zorlo pack) has no download record: checking it against the Hub would read all 18 GB
    if (r.ready && !fs::exists(dir / ".zorlo-pull.json", ec)) {
        say("\n  Using " + r.m.name + " from " + dir.u8string() + ".\n", { { "type", "note" }, { "text", "Using " + r.m.name + " from " + dir.u8string() + "." } });
    } else if (!r.m.repo.empty()) {
        if (r.ready) say("\n  Checking " + r.m.name + " for updates...\n", { { "type", "status" }, { "id", r.m.id }, { "text", "Checking " + r.m.name + " for updates" } });
        else {
            char gb[16];
            snprintf(gb, sizeof gb, "%.1f", r.m.gb);
            say("\n  Downloading " + r.m.name + " (" + gb + " GB). This can take a while.\n  If it stops, start Zorlo again: it continues where it left off.\n\n",
                { { "type", "status" }, { "id", r.m.id }, { "text", "Downloading " + r.m.name }, { "downloading", true } });
        }
        zorlo_pull_json(g_gui);
        if (zorlo_pull(r.m.repo, dir.u8string(), false, false, true) != 0) {
            if (!r.ready) {
                say("\n  The download did not finish. Check your internet connection and free disk space,\n  then start Zorlo again.\n",
                    { { "type", "error" }, { "text", "The download did not finish: " + zorlo_pull_last_error() + ". Check your internet connection and free disk space, then try again; it continues where it stopped." } });
                return 1;
            }
            say("  Could not check for updates (offline?). Starting the copy you have.\n", { { "type", "note" }, { "text", "Could not check for updates (offline?). Starting the copy you have." } });
        }
    }
    { std::ofstream o(models_dir / "last.txt"); o << r.m.id << "\n"; }

    say("\n  Starting " + r.m.name + ". Your browser opens the chat page when it is ready (about a minute).\n"
        "  Keep this window open while you chat. Close it to stop Zorlo.\n\n",
        { { "type", "starting" }, { "id", r.m.id }, { "name", r.m.name } });
    if (!g_gui && !getenv("ZORLO_NO_BROWSER")) open_browser_when_ready(8080);
    std::string pack = dir.u8string();
    // 12k tokens of context: the agent's instructions and tool list alone take ~2.5k (KV ~1.1 GB on Qwen3-30B-A3B)
    char a0[] = "zorlo", a1[] = "serve", a3[] = "-c", a4[] = "12288";
    char * sargv[] = { a0, a1, pack.data(), a3, a4 };
    const int rc = zorlo_serve_main(5, sargv);
    if (rc != 0) say("\n  Zorlo could not start. If it is already running in another window, use that one:\n  http://127.0.0.1:8080\n",
                     { { "type", "error" }, { "text", "Zorlo could not start the model. If Zorlo is already running, close it from its tray icon or with Task Manager, then try again." } });
    return rc;
}
#endif

