// Zorlo agent: the tools behind the agent app that `zorlo serve` opens at "/".
//
// The browser runs the agent loop (model call -> tool calls -> results -> model ...). Every tool runs here, in
// the server, and every permission decision is made here too, so the page can't bypass a setting:
//   POST /agent/tool {name, arguments, approved}  ->  ok | needs_approval | blocked | error
// Settings (%APPDATA%\Zorlo\settings.json): workspace folder, full permission or per-group allow/ask/deny
// (read, write, command, web, mcp), Brave Search key, skills folder, MCP servers (Claude Desktop format).
//
// Safety: the agent endpoints answer only the page this server served. Each launch makes a random token that
// is written into that page; requests must carry it, use Host 127.0.0.1/localhost (blocks DNS rebinding) and,
// when sent from a page, that same origin. Agent tools are off when serving on a non-loopback address.

#include "cli/agent.h"
#include "cli/extract_ps.h"
#include "cli/pull.h"
#include "httplib.h"  // before windows.h (winsock order)
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <winhttp.h>
#include <dpapi.h>
#pragma comment(lib, "crypt32.lib")
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const char * ZORLO_VERSION = "0.1";

// ---------- small helpers ----------
std::string env_or(const char * k, const std::string & d) { const char * v = getenv(k); return v && *v ? v : d; }

fs::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof buf / sizeof buf[0]));
    return fs::path(std::wstring(buf, n)).parent_path();
#else
    return fs::current_path();
#endif
}

std::string read_text(const fs::path & p, size_t max = SIZE_MAX) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p.u8string());
    std::string s;
    char buf[65536];
    while (s.size() < max) {
        in.read(buf, sizeof buf);
        if (in.gcount() <= 0) break;
        s.append(buf, (size_t) in.gcount());
    }
    if (s.size() > max) s.resize(max);
    return s;
}

std::string random_token() {
    std::random_device rd;
    std::string t;
    static const char * hex = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) t += hex[rd() & 15];
    return t;
}

std::string url_encode(const std::string & s) {
    static const char * hex = "0123456789ABCDEF";
    std::string o;
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char) c;
        else if (c == ' ') o += '+';
        else { o += '%'; o += hex[c >> 4]; o += hex[c & 15]; }
    }
    return o;
}

std::string url_decode(const std::string & s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && isxdigit((unsigned char) s[i + 1]) && isxdigit((unsigned char) s[i + 2])) {
            o += (char) std::stoi(s.substr(i + 1, 2), nullptr, 16); i += 2;
        } else o += s[i] == '+' ? ' ' : s[i];
    }
    return o;
}

std::string decode_entities(std::string s) {
    static const std::pair<const char *, const char *> ents[] = {
        { "&nbsp;", " " }, { "&lt;", "<" }, { "&gt;", ">" }, { "&quot;", "\"" }, { "&#39;", "'" }, { "&#x27;", "'" },
        { "&apos;", "'" }, { "&mdash;", "-" }, { "&ndash;", "-" }, { "&hellip;", "..." }, { "&amp;", "&" } };
    for (const auto & [e, r] : ents)
        for (size_t p; (p = s.find(e)) != std::string::npos;) s.replace(p, strlen(e), r);
    return s;
}

// readable text from a web page: drop scripts/styles, keep block breaks, strip tags
std::string html_to_text(const std::string & html, size_t max) {
    std::string s = std::regex_replace(html, std::regex("<(script|style|noscript|svg|head)[^>]*>[\\s\\S]*?</\\1>", std::regex::icase), " ");
    s = std::regex_replace(s, std::regex("<(br|/p|/div|/li|/h[1-6]|/tr|/section|/article)[^>]*>", std::regex::icase), "\n");
    s = std::regex_replace(s, std::regex("<li[^>]*>", std::regex::icase), "\n- ");
    s = std::regex_replace(s, std::regex("<[^>]+>"), " ");
    s = decode_entities(s);
    s = std::regex_replace(s, std::regex("[ \\t\\r\\f\\v]+"), " ");
    s = std::regex_replace(s, std::regex("\\n[ \\n]+"), "\n\n");
    if (s.size() > max) s = s.substr(0, max) + "\n[... page cut at " + std::to_string(max) + " characters]";
    return s;
}

std::string cut(const std::string & s, size_t max) {
    if (s.size() <= max) return s;
    return s.substr(0, max / 2) + "\n[... " + std::to_string(s.size() - max) + " characters cut ...]\n" + s.substr(s.size() - max / 2);
}

#ifdef _WIN32
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
std::string quote_arg(const std::string & a) {  // Windows command-line quoting (CommandLineToArgvW rules)
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string o = "\"";
    size_t bs = 0;
    for (char c : a) {
        if (c == '\\') { ++bs; continue; }
        if (c == '"') { o.append(bs * 2 + 1, '\\'); o += '"'; }
        else { o.append(bs, '\\'); o += c; }
        bs = 0;
    }
    o.append(bs * 2, '\\');
    return o + "\"";
}
#endif

// ---------- programs installed after Zorlo started (Node.js, uv) ----------
#ifdef _WIN32
// PATH as a newly started program would get it (system + user, from the registry), plus what Zorlo started with:
// Node.js or uv installed while Zorlo runs then work without restarting it
void refresh_path() {
    static const std::wstring start_path = [] { const wchar_t * p = _wgetenv(L"PATH"); return p ? std::wstring(p) : std::wstring(); }();
    auto read = [](HKEY root, const wchar_t * key) {
        std::vector<wchar_t> raw(32768), out(32768);
        DWORD sz = (DWORD) (raw.size() * sizeof(wchar_t));
        if (RegGetValueW(root, key, L"Path", RRF_RT_REG_SZ | RRF_RT_REG_EXPAND_SZ | RRF_NOEXPAND, nullptr, raw.data(), &sz) != ERROR_SUCCESS) return std::wstring();
        ExpandEnvironmentStringsW(raw.data(), out.data(), (DWORD) out.size());
        return std::wstring(out.data());
    };
    const std::wstring p = read(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment") + L";" +
                           read(HKEY_CURRENT_USER, L"Environment") + L";" + start_path;
    SetEnvironmentVariableW(L"PATH", p.c_str());
}
bool has_program(const wchar_t * exe) {
    wchar_t buf[MAX_PATH];
    return SearchPathW(nullptr, exe, nullptr, MAX_PATH, buf, nullptr) > 0;
}
#else
void refresh_path() {}
bool has_program(const wchar_t *) { return false; }
#endif
json runtimes() {
    refresh_path();
    return { { "node", has_program(L"npx.cmd") || has_program(L"npx") }, { "uv", has_program(L"uvx.exe") || has_program(L"uvx") } };
}

// ---------- MCP client (stdio, JSON-RPC 2.0, newline-delimited) ----------
#ifdef _WIN32
struct McpServer {
    std::string name, error;
    fs::path log;  // the connector's own error output, shown when it fails to start
    json tools = json::array();
    HANDLE proc = nullptr, job = nullptr, to_child = nullptr, from_child = nullptr;
    std::thread reader;
    std::mutex mu;
    std::condition_variable cv;
    std::map<int, json> answers;
    int next_id = 1;
    bool dead = false;

    ~McpServer() {
        if (job) CloseHandle(job);  // KILL_ON_JOB_CLOSE ends the server and its children
        if (to_child) CloseHandle(to_child);
        if (proc) { WaitForSingleObject(proc, 2000); CloseHandle(proc); }
        if (from_child) CloseHandle(from_child);
        if (reader.joinable()) reader.join();
    }

    void send(const json & msg) {
        const std::string line = msg.dump() + "\n";
        DWORD w = 0;
        if (!WriteFile(to_child, line.data(), (DWORD) line.size(), &w, nullptr)) throw std::runtime_error("connector stopped");
    }

    json request(const std::string & method, const json & params, int timeout_s) {
        int id;
        { std::lock_guard<std::mutex> lk(mu); id = next_id++; }
        send({ { "jsonrpc", "2.0" }, { "id", id }, { "method", method }, { "params", params } });
        std::unique_lock<std::mutex> lk(mu);
        if (!cv.wait_for(lk, std::chrono::seconds(timeout_s), [&] { return answers.count(id) || dead; }))
            throw std::runtime_error(method + ": no answer from connector " + name + " in " + std::to_string(timeout_s) + " s");
        if (!answers.count(id)) throw std::runtime_error("connector " + name + " stopped");
        json a = std::move(answers[id]);
        answers.erase(id);
        if (a.contains("error")) throw std::runtime_error(name + ": " + a["error"].value("message", a["error"].dump()));
        return a.value("result", json::object());
    }

    bool start(const json & cfg) {
        try {
            std::string cmdline = "cmd.exe /d /s /c \"" + quote_arg(cfg.at("command").get<std::string>());
            for (const auto & a : cfg.value("args", json::array())) cmdline += " " + quote_arg(a.get<std::string>());
            cmdline += "\"";
            SECURITY_ATTRIBUTES sa = { sizeof sa, nullptr, TRUE };
            HANDLE in_r, in_w, out_r, out_w;
            if (!CreatePipe(&in_r, &in_w, &sa, 0) || !CreatePipe(&out_r, &out_w, &sa, 0)) throw std::runtime_error("pipe failed");
            SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
            SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
            std::error_code lec;
            if (!log.empty()) fs::create_directories(log.parent_path(), lec);
            HANDLE nul = log.empty() ? INVALID_HANDLE_VALUE
                                     : CreateFileW(log.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, CREATE_ALWAYS, 0, nullptr);
            if (nul == INVALID_HANDLE_VALUE) nul = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
            // environment: ours plus the connector's "env"
            std::wstring envblock;
            if (cfg.contains("env") && cfg["env"].is_object()) {
                wchar_t * cur = GetEnvironmentStringsW();
                for (wchar_t * p = cur; *p; p += wcslen(p) + 1) envblock += std::wstring(p) + L'\0';
                FreeEnvironmentStringsW(cur);
                for (const auto & [k, v] : cfg["env"].items()) envblock += widen(k + "=" + v.get<std::string>()) + L'\0';
                envblock += L'\0';
            }
            STARTUPINFOW si = { sizeof si };
            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdInput = in_r; si.hStdOutput = out_w; si.hStdError = nul;
            PROCESS_INFORMATION pi;
            std::wstring wcmd = widen(cmdline);
            const BOOL ok = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE,
                                           CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | CREATE_SUSPENDED,
                                           envblock.empty() ? nullptr : envblock.data(), nullptr, &si, &pi);
            CloseHandle(in_r); CloseHandle(out_w); CloseHandle(nul);
            if (!ok) { CloseHandle(in_w); CloseHandle(out_r); throw std::runtime_error("cannot start: " + cfg.at("command").get<std::string>()); }
            job = CreateJobObjectW(nullptr, nullptr);
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
            AssignProcessToJobObject(job, pi.hProcess);
            ResumeThread(pi.hThread); CloseHandle(pi.hThread);
            proc = pi.hProcess; to_child = in_w; from_child = out_r;
            reader = std::thread([this] {
                std::string buf;
                char chunk[8192];
                DWORD n;
                const bool debug = getenv("ZORLO_MCP_DEBUG") != nullptr;
                while (ReadFile(from_child, chunk, sizeof chunk, &n, nullptr) && n) {
                    if (debug) fprintf(stderr, "[mcp %s] %.*s\n", name.c_str(), (int) n, chunk);
                    buf.append(chunk, n);
                    for (size_t p; (p = buf.find('\n')) != std::string::npos;) {
                        const std::string line = buf.substr(0, p);
                        buf.erase(0, p + 1);
                        try {
                            json m = json::parse(line);
                            if (m.contains("id") && m["id"].is_number_integer() && (m.contains("result") || m.contains("error"))) {
                                const int id = m["id"].get<int>();  // read before moving m: the move happens first in `a[m[..]] = move(m)`
                                std::lock_guard<std::mutex> lk(mu);
                                answers[id] = std::move(m);
                                cv.notify_all();
                            } else if (m.contains("id") && m.contains("method")) {  // a request to us: say we don't support it
                                send({ { "jsonrpc", "2.0" }, { "id", m["id"] }, { "error", { { "code", -32601 }, { "message", "not supported" } } } });
                            }
                        } catch (const std::exception & e) {  // logging lines on stdout: ignore
                            if (debug) fprintf(stderr, "[mcp %s] (not a message: %s)\n", name.c_str(), e.what());
                        }
                    }
                }
                std::lock_guard<std::mutex> lk(mu);
                dead = true;
                cv.notify_all();
            });
            request("initialize", { { "protocolVersion", "2025-06-18" }, { "capabilities", json::object() },
                                    { "clientInfo", { { "name", "zorlo" }, { "version", ZORLO_VERSION } } } }, 240);  // the first npx/uvx run downloads it
            send({ { "jsonrpc", "2.0" }, { "method", "notifications/initialized" } });
            tools = request("tools/list", json::object(), 60).value("tools", json::array());
            return true;
        } catch (const std::exception & e) {
            bool died;
            { std::lock_guard<std::mutex> lk(mu); died = dead; }
            error = died ? "stopped while starting" : e.what();
            // the last lines it printed usually say why (a missing key, Node.js not installed, ...)
            std::string tail;
            try { tail = read_text(log, 1 << 20); } catch (...) {}
            while (!tail.empty() && isspace((unsigned char) tail.back())) tail.pop_back();
            if (tail.size() > 600) tail = "..." + tail.substr(tail.size() - 600);
            if (!tail.empty()) error += ": " + tail;
            else if (died) error += " (check the command and its arguments)";
            return false;
        }
    }
};
#else
struct McpServer {
    std::string name, error = "connectors need Windows for now";
    fs::path log;
    json tools = json::array();
    bool start(const json &) { return false; }
    json request(const std::string &, const json &, int) { throw std::runtime_error(error); }
};
#endif

std::string tool_safe(std::string s) {
    for (char & c : s) if (!isalnum((unsigned char) c) && c != '_' && c != '-') c = '_';
    return s;
}

// ---------- state ----------
struct Agent {
    std::mutex mu;
    std::string token, host;
    int port = 8080;
    bool tools_on = true;
    fs::path settings_path, ui_dir;
    json settings;
    std::map<std::string, std::unique_ptr<McpServer>> mcp;
    std::map<std::string, std::pair<std::string, std::string>> mcp_tool;  // model-facing name -> (server, tool)
    bool mcp_started = false;
    bool mcp_starting = false;  // connectors are starting in the background
    json update;                // a newer Zorlo, from the online catalog (null when up to date)
};
Agent G;

fs::path appdata_dir() {
    fs::path base = fs::u8path(env_or("APPDATA", (exe_dir()).u8string()));
    return base / "Zorlo";
}

json default_settings() {
    const fs::path docs = fs::u8path(env_or("USERPROFILE", exe_dir().u8string())) / "Documents" / "Zorlo Projects";
    return {
        { "workspace", docs.u8string() },
        { "full_permission", false },
        { "permissions", { { "read", "allow" }, { "write", "ask" }, { "command", "ask" }, { "web", "allow" }, { "mcp", "ask" }, { "setup", "ask" } } },
        { "brave_api_key", "" },
        { "skills_dir", (appdata_dir() / "skills").u8string() },
        { "mcpServers", json::object() },
        { "thinking", false },
        { "mode", "exact" },
        { "max_steps", 25 },
        // the model the agent uses: Zorlo's own (this PC), or any OpenAI-compatible provider with the user's key
        { "provider", { { "kind", "zorlo" }, { "base_url", "" }, { "model", "" }, { "context", 128000 } } },
        { "provider_keys", json::object() },  // base URL -> API key; never sent to the page, only to that URL
    };
}

// ---------- secrets in settings.json: API keys and connector tokens, encrypted for this Windows user ----------
// Windows DPAPI: only the same user on the same PC can decrypt them; the file alone is useless to anyone else.
const char * B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
std::string b64_encode(const unsigned char * p, size_t n) {
    std::string o;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t) p[i] << 16 | (i + 1 < n ? (uint32_t) p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        o += B64[v >> 18 & 63]; o += B64[v >> 12 & 63];
        o += i + 1 < n ? B64[v >> 6 & 63] : '='; o += i + 2 < n ? B64[v & 63] : '=';
    }
    return o;
}
std::string b64_decode(const std::string & s) {
    std::string o;
    uint32_t v = 0;
    int bits = 0;
    for (char c : s) {
        const char * f = strchr(B64, c);
        if (c == '=' || !f || !c) continue;
        v = v << 6 | (uint32_t) (f - B64);
        if ((bits += 6) >= 8) { bits -= 8; o += (char) (v >> bits & 255); }
    }
    return o;
}
const std::string SECRET_TAG = "dpapi:";
std::string protect(const std::string & plain) {
#ifdef _WIN32
    if (plain.empty() || plain.rfind(SECRET_TAG, 0) == 0) return plain;
    DATA_BLOB in = { (DWORD) plain.size(), (BYTE *) plain.data() }, out = {};
    if (!CryptProtectData(&in, L"Zorlo", nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out)) return plain;
    std::string r = SECRET_TAG + b64_encode(out.pbData, out.cbData);
    LocalFree(out.pbData);
    return r;
#else
    return plain;
#endif
}
std::string unprotect(const std::string & stored) {
#ifdef _WIN32
    if (stored.rfind(SECRET_TAG, 0) != 0) return stored;  // written before encryption: used as is, encrypted on next save
    const std::string raw = b64_decode(stored.substr(SECRET_TAG.size()));
    DATA_BLOB in = { (DWORD) raw.size(), (BYTE *) raw.data() }, out = {};
    if (!CryptUnprotectData(&in, nullptr, nullptr, nullptr, nullptr, CRYPTPROTECT_UI_FORBIDDEN, &out)) return "";  // another user or PC
    std::string r((const char *) out.pbData, out.cbData);
    LocalFree(out.pbData);
    return r;
#else
    return stored;
#endif
}
// apply f to every secret: the Brave key, provider keys, and connectors' env values (tokens such as GITHUB_TOKEN)
void each_secret(json & s, const std::function<std::string(const std::string &)> & f) {
    if (s.contains("brave_api_key") && s["brave_api_key"].is_string()) s["brave_api_key"] = f(s["brave_api_key"].get<std::string>());
    if (s.contains("provider_keys") && s["provider_keys"].is_object())
        for (auto & [k, v] : s["provider_keys"].items()) if (v.is_string()) v = f(v.get<std::string>());
    if (s.contains("mcpServers") && s["mcpServers"].is_object())
        for (auto & [n, c] : s["mcpServers"].items())
            if (c.is_object() && c.contains("env") && c["env"].is_object())
                for (auto & [k, v] : c["env"].items()) if (v.is_string()) v = f(v.get<std::string>());
}

void save_settings() {
    fs::create_directories(G.settings_path.parent_path());
    const fs::path tmp = G.settings_path.u8string() + ".tmp";
    json disk = G.settings;
    each_secret(disk, protect);
    { std::ofstream o(tmp, std::ios::binary); o << disk.dump(2); }
    fs::rename(tmp, G.settings_path);
}

void load_settings() {
    G.settings = default_settings();
    bool plain_secrets = false;
    try {
        if (fs::exists(G.settings_path)) {
            const json s = json::parse(read_text(G.settings_path));
            for (const auto & [k, v] : s.items()) G.settings[k] = v;
            each_secret(G.settings, [&](const std::string & v) {
                if (!v.empty() && v.rfind(SECRET_TAG, 0) != 0) plain_secrets = true;
                return unprotect(v);
            });
        }
    } catch (...) {}  // a broken file falls back to defaults
    if (plain_secrets) try { save_settings(); } catch (...) {}  // a file from before encryption: encrypt it now
    std::error_code ec;
    fs::create_directories(fs::u8path(G.settings["workspace"].get<std::string>()), ec);
}

// a chat's own project folder: tool calls carry it, so a chat keeps working in its folder while the user looks at
// another chat in another folder (set for the length of one /agent/tool request)
thread_local std::string t_folder;
fs::path workspace() { return fs::u8path(t_folder.empty() ? G.settings.value("workspace", "") : t_folder); }

// a path the model gave, relative to the workspace; `inside` = stays within it
fs::path resolve(const std::string & p, bool & inside) {
    fs::path ws = fs::weakly_canonical(workspace());
    fs::path r = fs::u8path(p);
    if (r.is_relative()) r = ws / r;
    r = fs::weakly_canonical(r);
    const auto rel = r.lexically_relative(ws);
    inside = !rel.empty() && *rel.begin() != "..";
    if (r == ws) inside = true;
    return r;
}

// ---------- skills: <skills_dir>\<name>\SKILL.md with a "name:" / "description:" front matter ----------
void parse_front(const std::string & text, std::string & name, std::string & desc);  // below, with the skill installer
struct Skill { std::string name, description; fs::path file; };
std::vector<Skill> list_skills() {
    std::vector<Skill> out;
    std::error_code ec;
    const fs::path dir = fs::u8path(G.settings.value("skills_dir", ""));
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        const fs::path f = e.path() / "SKILL.md";
        if (!e.is_directory() || !fs::exists(f)) continue;
        Skill s{ e.path().filename().u8string(), "", f };
        try { parse_front(read_text(f, 16384), s.name, s.description); } catch (...) {}
        if (s.name.empty()) s.name = e.path().filename().u8string();
        out.push_back(s);
    }
    return out;
}

// ---------- MCP startup ----------
fs::path mcp_log(const std::string & name) { return appdata_dir() / "logs" / ("connector-" + tool_safe(name) + ".log"); }

void index_tools_locked(const std::string & name) {
    for (auto it = G.mcp_tool.begin(); it != G.mcp_tool.end();) it = it->second.first == name ? G.mcp_tool.erase(it) : std::next(it);
    const auto f = G.mcp.find(name);
    if (f == G.mcp.end() || !f->second->error.empty()) return;
    for (const auto & t : f->second->tools) {
        const std::string id = ("mcp__" + tool_safe(name) + "__" + tool_safe(t.value("name", ""))).substr(0, 64);
        G.mcp_tool[id] = { name, t.value("name", "") };
    }
}

// every connector that is switched on, started side by side in the background (npx may download for a minute):
// the app opens at once and the connectors' tools appear when they are ready. Called under G.mu.
void start_mcp_async_locked() {
    if (G.mcp_starting) return;
    G.mcp_starting = true;
    G.mcp_started = true;
    const json servers = G.settings.value("mcpServers", json::object());
    std::thread([servers] {
        refresh_path();
        std::vector<std::unique_ptr<McpServer>> list;
        std::vector<json> cfgs;
        for (const auto & [name, cfg] : servers.items()) {
            if (cfg.value("disabled", false)) continue;
            auto s = std::make_unique<McpServer>();
            s->name = name;
            s->log = mcp_log(name);
            list.push_back(std::move(s));
            cfgs.push_back(cfg);
        }
        std::vector<std::thread> th;
        for (size_t i = 0; i < list.size(); ++i) th.emplace_back([&, i] { list[i]->start(cfgs[i]); });
        for (auto & t : th) t.join();
        std::lock_guard<std::mutex> lk(G.mu);
        for (auto & s : list) { const std::string n = s->name; G.mcp[n] = std::move(s); }
        // drop any that were removed or switched off while these started
        const json now = G.settings.value("mcpServers", json::object());
        for (auto it = G.mcp.begin(); it != G.mcp.end();)
            it = !now.contains(it->first) || now[it->first].value("disabled", false) ? G.mcp.erase(it) : std::next(it);
        G.mcp_tool.clear();
        for (const auto & [n, s] : G.mcp) index_tools_locked(n);
        G.mcp_starting = false;
    }).detach();
}

json connector_status(const std::string & name) {  // under G.mu
    json r = { { "name", name } };
    const auto f = G.mcp.find(name);
    if (f == G.mcp.end()) { r["connected"] = false; r["error"] = "not running"; return r; }
    r["connected"] = f->second->error.empty();
    if (!f->second->error.empty()) r["error"] = f->second->error;
    json tools = json::array();
    for (const auto & t : f->second->tools) tools.push_back(t.value("name", ""));
    r["tools"] = tools;
    return r;
}

std::string clean_name(std::string n) {
    std::string o;
    for (char c : n) o += isalnum((unsigned char) c) || c == '-' || c == '_' ? (char) tolower((unsigned char) c) : (c == ' ' ? '-' : '\0');
    o.erase(std::remove(o.begin(), o.end(), '\0'), o.end());
    if (o.empty() || o.size() > 40) throw std::runtime_error("give the connector a short name (letters, digits, - or _)");
    return o;
}

// add or replace one connector and start it, without stopping the others; the slow start runs outside the lock
json connect_one(const std::string & name, json cfg) {
    if (!cfg.contains("command") || !cfg["command"].is_string() || cfg["command"].get<std::string>().empty())
        throw std::runtime_error("a connector needs a command (for example npx or uvx)");
    if (cfg.contains("args") && !cfg["args"].is_array()) throw std::runtime_error("args must be a list of strings");
    for (const auto & a : cfg.value("args", json::array())) if (!a.is_string()) throw std::runtime_error("args must be a list of strings");
    if (cfg.contains("env")) {
        if (!cfg["env"].is_object()) throw std::runtime_error("env must be an object of strings");
        for (const auto & [k, v] : cfg["env"].items()) if (!v.is_string()) throw std::runtime_error("env values must be strings");
    }
    cfg.erase("disabled");
    refresh_path();
    auto s = std::make_unique<McpServer>();
    s->name = name;
    s->log = mcp_log(name);
    s->start(cfg);
    std::lock_guard<std::mutex> lk(G.mu);
    G.settings["mcpServers"][name] = cfg;
    save_settings();
    if (!G.mcp_started) {  // the rest start on the next tool list; report this start
        json r = { { "name", name }, { "connected", s->error.empty() } };
        if (!s->error.empty()) r["error"] = s->error;
        return r;
    }
    G.mcp[name] = std::move(s);
    index_tools_locked(name);
    return connector_status(name);
}

void disconnect_locked(const std::string & name) {
    G.mcp.erase(name);  // stops its process
    index_tools_locked(name);
}

json remove_connector(const std::string & name) {
    std::lock_guard<std::mutex> lk(G.mu);
    if (!G.settings["mcpServers"].contains(name)) throw std::runtime_error("no connector named " + name);
    G.settings["mcpServers"].erase(name);
    save_settings();
    disconnect_locked(name);
    return { { "removed", name } };
}

json set_connector_enabled(const std::string & name, bool on) {
    json cfg;
    {
        std::lock_guard<std::mutex> lk(G.mu);
        if (!G.settings["mcpServers"].contains(name)) throw std::runtime_error("no connector named " + name);
        cfg = G.settings["mcpServers"][name];
        if (!on) {
            G.settings["mcpServers"][name]["disabled"] = true;
            save_settings();
            disconnect_locked(name);
            return { { "name", name }, { "enabled", false } };
        }
    }
    return connect_one(name, cfg);
}

// ---------- the connector market ----------
// A list of ready-made connectors: click (or ask the agent) to add one. Read from the Zorlo catalog on Hugging Face so
// new ones appear without a new Zorlo; this built-in copy is used offline. "{KEY}" in args/env is filled from fields.
const char * BUILTIN_MARKET = R"JSON({
"format": "zorlo-connectors-1",
"connectors": [
 {"id":"filesystem","name":"Files","by":"MCP project","category":"Files and code","runtime":"node","about":"Read, write and search files in one folder you choose (also outside the project).",
  "command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","{FOLDER}"],"fields":[{"key":"FOLDER","label":"Folder it may use","kind":"folder"}]},
 {"id":"git","name":"Git","by":"MCP project","category":"Files and code","runtime":"python","about":"Status, diffs, history, branches and commits of a git repository.",
  "command":"uvx","args":["mcp-server-git","--repository","{REPO}"],"fields":[{"key":"REPO","label":"Repository folder","kind":"folder"}]},
 {"id":"github","name":"GitHub","by":"GitHub","category":"Files and code","runtime":"node","about":"Issues, pull requests, code search and repositories on GitHub (GitHub's own server).",
  "command":"npx","args":["-y","mcp-remote","https://api.githubcopilot.com/mcp/","--header","Authorization:${AUTH_HEADER}"],"env":{"AUTH_HEADER":"Bearer {TOKEN}"},
  "fields":[{"key":"TOKEN","label":"GitHub personal access token","kind":"secret","help":"https://github.com/settings/personal-access-tokens"}]},
 {"id":"memory","name":"Memory","by":"MCP project","category":"Thinking","runtime":"node","about":"A small knowledge graph the agent can store facts in and recall in later chats.",
  "command":"npx","args":["-y","@modelcontextprotocol/server-memory"]},
 {"id":"sequential-thinking","name":"Sequential thinking","by":"MCP project","category":"Thinking","runtime":"node","about":"Lets the agent work through hard problems step by step, revising as it goes.",
  "command":"npx","args":["-y","@modelcontextprotocol/server-sequential-thinking"]},
 {"id":"time","name":"Time","by":"MCP project","category":"Thinking","runtime":"python","about":"Current time and time-zone conversions.",
  "command":"uvx","args":["mcp-server-time"]},
 {"id":"playwright","name":"Playwright browser","by":"Microsoft","category":"Browser","runtime":"node","about":"Opens a real browser: click, type, fill forms, take snapshots of web pages.",
  "command":"npx","args":["-y","@playwright/mcp@latest"]},
 {"id":"chrome-devtools","name":"Chrome DevTools","by":"Google","category":"Browser","runtime":"node","about":"Drives Chrome and reads its console, network and performance traces: debug websites.",
  "command":"npx","args":["-y","chrome-devtools-mcp@latest"]},
 {"id":"fetch","name":"Fetch","by":"MCP project","category":"Web","runtime":"python","about":"Downloads web pages and turns them into Markdown.",
  "command":"uvx","args":["mcp-server-fetch"]},
 {"id":"context7","name":"Context7 docs","by":"Upstash","category":"Web","runtime":"node","about":"Up-to-date documentation and code examples for thousands of libraries.",
  "command":"npx","args":["-y","@upstash/context7-mcp"]},
 {"id":"brave-search","name":"Brave Search","by":"Brave","category":"Web","runtime":"node","about":"Web, news, image and video search.",
  "command":"npx","args":["-y","@brave/brave-search-mcp-server"],"env":{"BRAVE_API_KEY":"{KEY}"},
  "fields":[{"key":"KEY","label":"Brave Search API key","kind":"secret","help":"https://brave.com/search/api/"}]},
 {"id":"tavily","name":"Tavily","by":"Tavily","category":"Web","runtime":"node","about":"Search and page extraction made for AI agents.",
  "command":"npx","args":["-y","tavily-mcp@latest"],"env":{"TAVILY_API_KEY":"{KEY}"},
  "fields":[{"key":"KEY","label":"Tavily API key","kind":"secret","help":"https://app.tavily.com/"}]},
 {"id":"exa","name":"Exa","by":"Exa","category":"Web","runtime":"node","about":"Neural web search, code search and company research.",
  "command":"npx","args":["-y","exa-mcp-server"],"env":{"EXA_API_KEY":"{KEY}"},
  "fields":[{"key":"KEY","label":"Exa API key","kind":"secret","help":"https://dashboard.exa.ai/api-keys"}]},
 {"id":"firecrawl","name":"Firecrawl","by":"Firecrawl","category":"Web","runtime":"node","about":"Scrape and crawl websites, including pages that need JavaScript.",
  "command":"npx","args":["-y","firecrawl-mcp"],"env":{"FIRECRAWL_API_KEY":"{KEY}"},
  "fields":[{"key":"KEY","label":"Firecrawl API key","kind":"secret","help":"https://www.firecrawl.dev/app/api-keys"}]},
 {"id":"notion","name":"Notion","by":"Notion","category":"Work apps","runtime":"node","about":"Search, read and edit pages and databases in your Notion workspace.",
  "command":"npx","args":["-y","@notionhq/notion-mcp-server"],"env":{"NOTION_TOKEN":"{TOKEN}"},
  "fields":[{"key":"TOKEN","label":"Notion integration token","kind":"secret","help":"https://www.notion.so/profile/integrations"}]},
 {"id":"linear","name":"Linear","by":"Linear","category":"Work apps","runtime":"node","about":"Find, create and update Linear issues and projects. You sign in in your browser.",
  "command":"npx","args":["-y","mcp-remote","https://mcp.linear.app/mcp"]},
 {"id":"sentry","name":"Sentry","by":"Sentry","category":"Work apps","runtime":"node","about":"Look into errors and issues from your Sentry projects. You sign in in your browser.",
  "command":"npx","args":["-y","mcp-remote","https://mcp.sentry.dev/mcp"]},
 {"id":"figma","name":"Figma","by":"Framelink","category":"Design","runtime":"node","about":"Reads Figma files and frames so the agent can build them as code.",
  "command":"npx","args":["-y","figma-developer-mcp","--stdio"],"env":{"FIGMA_API_KEY":"{KEY}"},
  "fields":[{"key":"KEY","label":"Figma personal access token","kind":"secret","help":"https://www.figma.com/developers/api#access-tokens"}]},
 {"id":"supabase","name":"Supabase","by":"Supabase","category":"Data","runtime":"node","about":"Manage Supabase projects: tables, SQL queries, migrations, logs.",
  "command":"npx","args":["-y","@supabase/mcp-server-supabase@latest"],"env":{"SUPABASE_ACCESS_TOKEN":"{TOKEN}"},
  "fields":[{"key":"TOKEN","label":"Supabase access token","kind":"secret","help":"https://supabase.com/dashboard/account/tokens"}]},
 {"id":"shopify-dev","name":"Shopify Dev","by":"Shopify","category":"Data","runtime":"node","about":"Shopify developer docs and API schemas for building apps and themes.",
  "command":"npx","args":["-y","@shopify/dev-mcp@latest"]},
 {"id":"netlify","name":"Netlify","by":"Netlify","category":"Data","runtime":"node","about":"Create, deploy and manage Netlify sites.",
  "command":"npx","args":["-y","@netlify/mcp"]}
],
"skill_sources": [ {"repo":"anthropics/skills","path":"skills","branch":"HEAD","by":"Anthropic","license":"Apache-2.0"} ]
})JSON";
const char * MARKET_URL = "https://huggingface.co/datasets/Haidershahi1/zorlo-catalog/resolve/main/connectors.json";

json market() {
    static std::mutex mu;
    static json cache;
    static std::chrono::steady_clock::time_point at;
    std::lock_guard<std::mutex> lk(mu);
    if (!cache.is_null() && std::chrono::steady_clock::now() - at < std::chrono::hours(1)) return cache;
    json m;
    std::string text, err;
    const std::string src = env_or("ZORLO_CONNECTORS", MARKET_URL);
    try {
        bool ok;
        if (src.rfind("https://", 0) == 0) ok = zorlo_fetch_text(src, text, err);
        else { text = read_text(fs::u8path(src)); ok = true; }
        if (ok) {
            if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
            m = json::parse(text);
            if (m.value("format", "") != "zorlo-connectors-1" || !m.contains("connectors")) m = nullptr;
        }
    } catch (...) { m = nullptr; }
    if (m.is_null()) m = json::parse(BUILTIN_MARKET);
    cache = m;
    at = std::chrono::steady_clock::now();
    return cache;
}

json market_entry(const std::string & id) {
    for (const auto & e : market().value("connectors", json::array())) if (e.value("id", "") == id) return e;
    throw std::runtime_error("no connector \"" + id + "\" in the list; use find_connectors to see what there is");
}

// the connector's command with the user's values filled in
json fill_entry(const json & e, const json & values) {
    auto fill = [&](std::string s) {
        for (const auto & f : e.value("fields", json::array())) {
            const std::string key = f.value("key", ""), ph = "{" + key + "}";
            if (s.find(ph) == std::string::npos) continue;
            std::string v = values.contains(key) && values[key].is_string() ? values[key].get<std::string>() : "";
            v.erase(0, v.find_first_not_of(" \t\r\n"));
            v.erase(v.find_last_not_of(" \t\r\n") + 1);
            if (v.empty()) throw std::runtime_error("needs " + f.value("label", key) + " (\"" + key + "\")");
            if (f.value("kind", "") == "folder") {
                std::error_code ec;
                if (!fs::is_directory(fs::u8path(v), ec)) throw std::runtime_error(f.value("label", key) + ": no such folder: " + v);
            }
            for (size_t p; (p = s.find(ph)) != std::string::npos;) s.replace(p, ph.size(), v);
        }
        return s;
    };
    json cfg = { { "command", e.at("command") }, { "args", json::array() }, { "market", e.value("id", "") } };
    for (const auto & a : e.value("args", json::array())) cfg["args"].push_back(fill(a.get<std::string>()));
    if (e.contains("env")) for (const auto & [k, v] : e["env"].items()) cfg["env"][k] = fill(v.get<std::string>());
    return cfg;
}

json add_connector(const json & a) {
    auto s = [&](const char * k) { return a.contains(k) && a[k].is_string() ? a[k].get<std::string>() : std::string(); };
    if (!s("id").empty()) {
        const json e = market_entry(s("id"));
        const std::string rt = e.value("runtime", "");
        const json have = runtimes();
        if (rt == "node" && !have["node"].get<bool>())
            throw std::runtime_error("this connector needs Node.js, which is not installed. Install it from https://nodejs.org (LTS) "
                                     "or with: winget install OpenJS.NodeJS.LTS, then add the connector again");
        if (rt == "python" && !have["uv"].get<bool>())
            throw std::runtime_error("this connector needs uv (Python), which is not installed. Install it with: "
                                     "winget install astral-sh.uv, then add the connector again");
        return connect_one(clean_name(s("name").empty() ? e.value("id", "") : s("name")), fill_entry(e, a.value("values", json::object())));
    }
    if (!s("url").empty()) {  // a remote (HTTP) MCP server, through the mcp-remote bridge
        const std::string url = s("url");
        if (url.rfind("https://", 0) != 0 && url.rfind("http://127.0.0.1", 0) != 0 && url.rfind("http://localhost", 0) != 0)
            throw std::runtime_error("the server address must start with https://");
        if (!runtimes()["node"].get<bool>()) throw std::runtime_error("remote connectors need Node.js (https://nodejs.org)");
        return connect_one(clean_name(s("name").empty() ? "remote" : s("name")), { { "command", "npx" }, { "args", { "-y", "mcp-remote", url } } });
    }
    if (s("command").empty()) throw std::runtime_error("give either id (from find_connectors), url (a remote server), or command (+ args, env)");
    json cfg = { { "command", s("command") }, { "args", a.value("args", json::array()) } };
    if (a.contains("env")) cfg["env"] = a["env"];
    return connect_one(clean_name(s("name").empty() ? s("command") : s("name")), cfg);
}

// ---------- skills from GitHub ----------
void parse_front(const std::string & text, std::string & name, std::string & desc) {
    auto trim = [](std::string & x) { x.erase(0, x.find_first_not_of(" \t\"'")); x.erase(x.find_last_not_of(" \t\r\"'") + 1); };
    std::istringstream in(text);
    std::string line;
    if (!std::getline(in, line) || line.rfind("---", 0) != 0) return;
    std::string * block = nullptr;  // a "key: >" or "key: |" value continues on the indented lines below
    while (std::getline(in, line) && line.rfind("---", 0) != 0) {
        if (block && !line.empty() && (line[0] == ' ' || line[0] == '\t')) {
            std::string part = line;
            trim(part);
            if (!part.empty()) *block += (block->empty() ? "" : " ") + part;
            continue;
        }
        block = nullptr;
        std::string * target = line.rfind("name:", 0) == 0 ? &name : line.rfind("description:", 0) == 0 ? &desc : nullptr;
        if (!target) continue;
        std::string v = line.substr(line.find(':') + 1);
        trim(v);
        if (v == ">" || v == "|" || v == ">-" || v == "|-" || v == ">+" || v == "|+") { target->clear(); block = target; }
        else *target = v;
    }
    trim(name); trim(desc);
}

std::string path_encode(const std::string & p) {  // a repository path in a URL: keep '/', encode the rest
    std::string o;
    for (const std::string & part : [&] { std::vector<std::string> v; std::string c; std::istringstream in(p); while (std::getline(in, c, '/')) v.push_back(c); return v; }()) {
        if (!o.empty()) o += '/';
        for (unsigned char c : part) {
            if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char) c;
            else { static const char * hex = "0123456789ABCDEF"; o += '%'; o += hex[c >> 4]; o += hex[c & 15]; }
        }
    }
    return o;
}

struct GhRef { std::string owner, repo, branch = "HEAD", path; };
GhRef parse_github(std::string u) {
    while (!u.empty() && (u.back() == '/' || isspace((unsigned char) u.back()))) u.pop_back();
    u.erase(0, u.find_first_not_of(" \t"));
    std::vector<std::string> p;
    auto split = [&](const std::string & s) { std::string c; std::istringstream in(s); while (std::getline(in, c, '/')) if (!c.empty()) p.push_back(c); };
    GhRef r;
    bool raw = false;
    if (u.rfind("https://github.com/", 0) == 0) split(u.substr(19));
    else if (u.rfind("https://raw.githubusercontent.com/", 0) == 0) { split(u.substr(34)); raw = true; }
    else if (u.find("://") == std::string::npos) split(u);  // owner/repo[/path]
    else throw std::runtime_error("give a GitHub link (https://github.com/owner/repo/tree/main/path-to-skill)");
    if (p.size() < 2) throw std::runtime_error("that is not a GitHub repository link");
    r.owner = p[0]; r.repo = p[1];
    if (r.repo.size() > 4 && r.repo.compare(r.repo.size() - 4, 4, ".git") == 0) r.repo.resize(r.repo.size() - 4);
    size_t i = 2;
    if (raw && p.size() > 2) { r.branch = p[2]; i = 3; }
    else if (p.size() > 3 && (p[2] == "tree" || p[2] == "blob")) { r.branch = p[3]; i = 4; }
    for (; i < p.size(); ++i) r.path += (r.path.empty() ? "" : "/") + url_decode(p[i]);
    if (r.path == "SKILL.md") r.path.clear();
    else if (r.path.size() > 9 && r.path.compare(r.path.size() - 9, 9, "/SKILL.md") == 0) r.path.resize(r.path.size() - 9);
    return r;
}

json gh_tree(const GhRef & r) {  // every file of the repository (one GitHub API call, cached 10 minutes)
    static std::mutex mu;
    static std::map<std::string, std::pair<std::chrono::steady_clock::time_point, json>> cache;
    const std::string key = r.owner + "/" + r.repo + "@" + r.branch;
    {
        std::lock_guard<std::mutex> lk(mu);
        const auto f = cache.find(key);
        if (f != cache.end() && std::chrono::steady_clock::now() - f->second.first < std::chrono::minutes(10)) return f->second.second;
    }
    std::string body, err;
    if (!zorlo_http_get("https://api.github.com/repos/" + path_encode(r.owner) + "/" + path_encode(r.repo) + "/git/trees/" + path_encode(r.branch) + "?recursive=1",
                        "Accept: application/vnd.github+json\r\n", body, err, 32 << 20))
        throw std::runtime_error("GitHub: " + err);
    json t = json::parse(body).value("tree", json::array());
    std::lock_guard<std::mutex> lk(mu);
    cache[key] = { std::chrono::steady_clock::now(), t };
    return t;
}

fs::path skills_dir() { return fs::u8path(G.settings.value("skills_dir", (appdata_dir() / "skills").u8string())); }

std::string safe_folder(std::string n) {
    for (char & c : n) if (!isalnum((unsigned char) c) && c != '-' && c != '_' && c != '.') c = '-';
    while (!n.empty() && n[0] == '.') n.erase(0, 1);
    if (n.empty() || n.size() > 64) throw std::runtime_error("bad skill name");
    return n;
}

json install_skill_github(const std::string & url) {
    GhRef r = parse_github(url);
    const json tree = gh_tree(r);
    const std::string pre = r.path.empty() ? "" : r.path + "/";
    bool here = false;
    std::vector<std::string> below;
    for (const auto & e : tree) {
        const std::string p = e.value("path", "");
        if (p == pre + "SKILL.md") here = true;
        else if (p.rfind(pre, 0) == 0 && p.size() > 9 && p.compare(p.size() - 9, 9, "/SKILL.md") == 0) below.push_back(p.substr(0, p.size() - 9));
    }
    if (!here) {
        if (below.empty()) throw std::runtime_error("no SKILL.md in " + r.owner + "/" + r.repo + (r.path.empty() ? "" : "/" + r.path) + ": that is not a skill");
        if (below.size() > 1) {
            std::string names;
            for (size_t i = 0; i < below.size() && i < 40; ++i) names += (i ? ", " : "") + below[i].substr(below[i].rfind('/') + 1);
            throw std::runtime_error("this folder has " + std::to_string(below.size()) + " skills; install one by its link (" +
                                     "https://github.com/" + r.owner + "/" + r.repo + "/tree/" + r.branch + "/" + below[0] + "). Skills here: " + names);
        }
        r.path = below[0];
    }
    const std::string base = r.path.empty() ? "" : r.path + "/";
    std::vector<std::pair<std::string, uint64_t>> files;
    uint64_t total = 0;
    for (const auto & e : tree) {
        const std::string p = e.value("path", "");
        if (e.value("type", "") != "blob" || p.rfind(base, 0) != 0) continue;
        const std::string rel = p.substr(base.size());
        if (rel.find("..") != std::string::npos || rel.find(':') != std::string::npos || rel.find('\\') != std::string::npos) continue;
        if (r.path.empty() && rel.find('/') != std::string::npos && rel.rfind("scripts/", 0) != 0 && rel.rfind("references/", 0) != 0 &&
            rel.rfind("assets/", 0) != 0) continue;  // a whole repository that is one skill: only the skill's own folders
        files.push_back({ rel, e.value("size", (uint64_t) 0) });
        total += files.back().second;
    }
    if (files.size() > 400 || total > (40u << 20)) throw std::runtime_error("that skill is too big (" + std::to_string(files.size()) + " files)");
    const std::string folder = safe_folder(r.path.empty() ? r.repo : r.path.substr(r.path.rfind('/') + 1));
    const fs::path dir = skills_dir(), tmp = dir / ("." + folder + ".download"), dst = dir / folder;
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);
    for (const auto & [rel, size] : files) {
        std::string body, err;
        if (!zorlo_http_get("https://raw.githubusercontent.com/" + path_encode(r.owner) + "/" + path_encode(r.repo) + "/" + path_encode(r.branch) + "/" +
                                path_encode(base + rel), "", body, err, (size_t) size + 1024))
            { fs::remove_all(tmp, ec); throw std::runtime_error("download of " + rel + " failed: " + err); }
        const fs::path f = tmp / fs::u8path(rel);
        fs::create_directories(f.parent_path());
        std::ofstream(f, std::ios::binary) << body;
    }
    fs::remove_all(dst, ec);  // an update replaces the old copy
    fs::rename(tmp, dst);
    std::string name = folder, desc;
    try { parse_front(read_text(dst / "SKILL.md", 16384), name, desc); } catch (...) {}
    return { { "installed", name }, { "folder", dst.u8string() }, { "files", files.size() }, { "description", desc },
             { "source", "https://github.com/" + r.owner + "/" + r.repo + (r.path.empty() ? "" : "/tree/" + r.branch + "/" + r.path) } };
}

json install_skill(const json & a) {
    auto s = [&](const char * k) { return a.contains(k) && a[k].is_string() ? a[k].get<std::string>() : std::string(); };
    if (!s("url").empty()) return install_skill_github(s("url"));
    if (s("name").empty() || s("instructions").empty())
        throw std::runtime_error("give url (a GitHub link to a skill folder) or name + description + instructions to write a new skill");
    const std::string folder = safe_folder(s("name"));
    const fs::path dir = skills_dir() / folder;
    fs::create_directories(dir);
    std::string desc = s("description");
    for (char & c : desc) if (c == '\n' || c == '\r') c = ' ';
    std::ofstream(dir / "SKILL.md", std::ios::binary) << "---\nname: " << folder << "\ndescription: " << desc << "\n---\n\n" << s("instructions") << "\n";
    return { { "installed", folder }, { "folder", dir.u8string() } };
}

json remove_skill(const std::string & name) {
    const std::string folder = safe_folder(name);
    const fs::path d = skills_dir() / folder;
    std::error_code ec;
    if (!fs::exists(d / "SKILL.md", ec)) throw std::runtime_error("no skill named " + name);
    fs::remove_all(d, ec);
    if (ec) throw std::runtime_error("could not delete " + d.u8string() + ": " + ec.message());
    return { { "removed", folder } };
}

// skills offered by the sources in the market (e.g. github.com/anthropics/skills), with their descriptions
json skill_market() {
    static std::mutex mu;
    static json cache;
    static std::chrono::steady_clock::time_point at;
    std::lock_guard<std::mutex> lk(mu);
    if (!cache.is_null() && std::chrono::steady_clock::now() - at < std::chrono::hours(1)) return cache;
    json out = json::array();
    std::string errors;
    for (const auto & src : market().value("skill_sources", json::array())) {
        try {
            GhRef r = parse_github(src.value("repo", ""));
            r.branch = src.value("branch", "HEAD");
            const std::string pre = src.value("path", "").empty() ? "" : src.value("path", "") + "/";
            std::vector<std::string> dirs;
            for (const auto & e : gh_tree(r)) {
                const std::string p = e.value("path", "");
                if (p.rfind(pre, 0) == 0 && p.size() > pre.size() + 9 && p.compare(p.size() - 9, 9, "/SKILL.md") == 0 &&
                    p.find('/', pre.size()) == p.size() - 9) dirs.push_back(p.substr(0, p.size() - 9));
            }
            std::vector<json> items(dirs.size());
            std::vector<std::thread> th;
            for (size_t i = 0; i < dirs.size(); ++i)
                th.emplace_back([&, i] {
                    std::string body, err, name = dirs[i].substr(dirs[i].rfind('/') + 1), desc;
                    if (zorlo_http_get("https://raw.githubusercontent.com/" + r.owner + "/" + r.repo + "/" + r.branch + "/" + path_encode(dirs[i]) + "/SKILL.md",
                                       "", body, err, 16384))
                        parse_front(body, name, desc);
                    items[i] = { { "name", name }, { "folder", dirs[i].substr(dirs[i].rfind('/') + 1) }, { "description", desc },
                                 { "by", src.value("by", r.owner) }, { "license", src.value("license", "") },
                                 { "url", "https://github.com/" + r.owner + "/" + r.repo + "/tree/" + r.branch + "/" + dirs[i] } };
                });
            for (auto & t : th) t.join();
            for (auto & it : items) out.push_back(it);
        } catch (const std::exception & e) { errors += std::string(e.what()) + " "; }
    }
    if (out.empty() && !errors.empty()) throw std::runtime_error(errors);
    cache = out;
    at = std::chrono::steady_clock::now();
    return cache;
}

bool matches(const json & e, const std::string & q) {
    if (q.empty()) return true;
    std::string hay = e.dump(), ql = q;
    std::transform(hay.begin(), hay.end(), hay.begin(), ::tolower);
    std::transform(ql.begin(), ql.end(), ql.begin(), ::tolower);
    std::istringstream in(ql);
    for (std::string w; in >> w;) if (hay.find(w) != std::string::npos) return true;
    return false;
}

// ---------- tool definitions the model sees ----------
json fn(const std::string & name, const std::string & desc, const json & props, const std::vector<std::string> & req) {
    return { { "type", "function" },
             { "function", { { "name", name }, { "description", desc },
                             { "parameters", { { "type", "object" }, { "properties", props }, { "required", req } } } } } };
}

json tool_list_locked() {
    json t = json::array();
    t.push_back(fn("list_dir", "List the files and folders in a folder of the project (path relative to the project folder; \".\" = the project folder). "
                   "depth 2-4 shows subfolders as a tree.",
                   { { "path", { { "type", "string" } } }, { "depth", { { "type", "integer" }, { "description", "1 (default) to 4" } } } }, { "path" }));
    t.push_back(fn("search_files", "Search the text of the project's files. Returns file:line: text for each match. "
                   "Plain text match ignores case; set regex true for a regular expression. glob limits file names, e.g. \"*.js\".",
                   { { "query", { { "type", "string" } } }, { "path", { { "type", "string" } } }, { "glob", { { "type", "string" } } },
                     { "regex", { { "type", "boolean" } } } }, { "query" }));
    t.push_back(fn("read_file", "Read a text file. Returns numbered lines. Use offset/limit (line numbers) for long files.",
                   { { "path", { { "type", "string" } } }, { "offset", { { "type", "integer" } } }, { "limit", { { "type", "integer" } } } }, { "path" }));
    t.push_back(fn("read_document", "Read a file read_file can't: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), OpenDocument, "
                   "pictures (the words in them, by OCR; you can't see images) and video/audio (length and size only). Returns its text.",
                   { { "path", { { "type", "string" } } }, { "offset", { { "type", "integer" }, { "description", "character to start at, for long documents" } } } },
                   { "path" }));
    t.push_back(fn("write_file", "Create a file or replace its whole content. Folders are created as needed.",
                   { { "path", { { "type", "string" } } }, { "content", { { "type", "string" } } } }, { "path", "content" }));
    t.push_back(fn("edit_file", "Replace one exact piece of text in a file. old_text must appear exactly once; include enough surrounding lines to make it unique.",
                   { { "path", { { "type", "string" } } }, { "old_text", { { "type", "string" } } }, { "new_text", { { "type", "string" } } } },
                   { "path", "old_text", "new_text" }));
    t.push_back(fn("run_command", "Run a Windows PowerShell command in the project folder and return its output and exit code. "
                   "Don't start servers or programs that keep running (they are stopped at the timeout); websites are previewed with preview_url.",
                   { { "command", { { "type", "string" } } }, { "timeout_s", { { "type", "integer" }, { "description", "default 120, max 1800" } } } },
                   { "command" }));
    t.push_back(fn("web_search", "Search the web. Returns titles, links and short snippets.", { { "query", { { "type", "string" } } } }, { "query" }));
    t.push_back(fn("fetch_url", "Download a web page and return its readable text (or raw text for JSON and plain text).",
                   { { "url", { { "type", "string" } } } }, { "url" }));
    t.push_back(fn("preview_url", "Get the address where the user can open an HTML file of the project in their browser.",
                   { { "path", { { "type", "string" } } } }, { "path" }));
    t.push_back(fn("find_connectors", "Search Zorlo's list of ready-made connectors (MCP servers: GitHub, Notion, a browser, databases, search, ...) "
                   "that give you new tools. Returns each one's id, what it needs (API keys, folders) and whether it is installed.",
                   { { "query", { { "type", "string" }, { "description", "words to look for; empty lists all" } } } }, {}));
    t.push_back(fn("add_connector", "Add and start a connector, so its tools are yours from the next step on. Use id from find_connectors with values for "
                   "its fields (ask the user for keys or tokens first; never invent them). Or url for a remote MCP server, or command + args + env "
                   "for any other MCP server.",
                   { { "id", { { "type", "string" } } }, { "values", { { "type", "object" }, { "description", "field key -> value" } } },
                     { "name", { { "type", "string" }, { "description", "optional short name" } } }, { "url", { { "type", "string" } } },
                     { "command", { { "type", "string" } } }, { "args", { { "type", "array" }, { "items", { { "type", "string" } } } } },
                     { "env", { { "type", "object" } } } }, {}));
    t.push_back(fn("remove_connector", "Remove a connector by its name.", { { "name", { { "type", "string" } } } }, { "name" }));
    t.push_back(fn("find_skills", "Search the skills that can be installed (instruction packs for tasks like PDFs, Word or PowerPoint files, "
                   "web design, testing websites, building MCP servers). Returns name, description, link and whether it is installed.",
                   { { "query", { { "type", "string" } } } }, {}));
    t.push_back(fn("install_skill", "Install a skill: url = a GitHub link to a skill folder (from find_skills or the user). Or write a new skill "
                   "yourself with name, description (when to use it) and instructions (Markdown).",
                   { { "url", { { "type", "string" } } }, { "name", { { "type", "string" } } }, { "description", { { "type", "string" } } },
                     { "instructions", { { "type", "string" } } } }, {}));
    const auto skills = list_skills();
    if (!skills.empty())
        t.push_back(fn("load_skill", "Load the instructions of a skill (listed in the system prompt) before doing a task it covers.",
                       { { "name", { { "type", "string" } } } }, { "name" }));
    if (!G.mcp_started) start_mcp_async_locked();
    // Connector tools are not listed one by one: their full descriptions would fill much of the model's 12K context
    // before a chat starts. The system prompt names them; these two read a tool's details and call it.
    if (!G.mcp_tool.empty()) {
        t.push_back(fn("connector_tools", "Show the tools of a connector (listed in the system prompt) with what each does and the arguments it takes. "
                       "Call this before use_connector the first time you use a tool.",
                       { { "connector", { { "type", "string" } } }, { "tool", { { "type", "string" }, { "description", "optional: just this tool" } } } }, { "connector" }));
        t.push_back(fn("use_connector", "Call a connector's tool. arguments must match that tool's inputSchema from connector_tools.",
                       { { "connector", { { "type", "string" } } }, { "tool", { { "type", "string" } } }, { "arguments", { { "type", "object" } } } },
                       { "connector", "tool" }));
    }
    return t;
}

std::string group_of(const std::string & tool) {
    if (tool == "list_dir" || tool == "read_file" || tool == "preview_url" || tool == "search_files" || tool == "read_document") return "read";
    if (tool == "write_file" || tool == "edit_file") return "write";
    if (tool == "run_command") return "command";
    if (tool == "web_search" || tool == "fetch_url") return "web";
    if (tool == "load_skill") return "skills";
    if (tool == "find_connectors" || tool == "find_skills") return "web";
    if (tool == "add_connector" || tool == "remove_connector" || tool == "install_skill") return "setup";
    if (tool.rfind("mcp__", 0) == 0 || tool == "use_connector") return "mcp";
    if (tool == "connector_tools") return "read";
    return "";
}

// ---------- the tools ----------
#ifdef _WIN32
json run_command(const std::string & command, int timeout_s) {
    // PowerShell, UTF-8 output, in the project folder; a job object stops the whole process tree at the timeout
    const std::string script = "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                               "$OutputEncoding=[Text.Encoding]::UTF8; " + command;
    const std::wstring wscript = widen(script);
    std::string b64;
    {
        const unsigned char * p = (const unsigned char *) wscript.data();
        const size_t n = wscript.size() * 2;
        static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (size_t i = 0; i < n; i += 3) {
            const uint32_t v = (uint32_t) p[i] << 16 | (i + 1 < n ? (uint32_t) p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
            b64 += tbl[v >> 18 & 63]; b64 += tbl[v >> 12 & 63];
            b64 += i + 1 < n ? tbl[v >> 6 & 63] : '='; b64 += i + 2 < n ? tbl[v & 63] : '=';
        }
    }
    std::wstring cmd = L"powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " + widen(b64);
    SECURITY_ATTRIBUTES sa = { sizeof sa, nullptr, TRUE };
    HANDLE out_r, out_w;
    CreatePipe(&out_r, &out_w, &sa, 0);
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul_in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si = { sizeof si };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul_in; si.hStdOutput = out_w; si.hStdError = out_w;
    PROCESS_INFORMATION pi;
    const std::wstring cwd = workspace().wstring();
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                                   cwd.c_str(), &si, &pi);
    CloseHandle(out_w); CloseHandle(nul_in);
    if (!ok) { CloseHandle(out_r); throw std::runtime_error("cannot start PowerShell"); }
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
    li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof li);
    AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread); CloseHandle(pi.hThread);

    std::string out;
    std::thread rd([&] {
        char buf[8192];
        DWORD n;
        while (ReadFile(out_r, buf, sizeof buf, &n, nullptr) && n) {
            out.append(buf, n);
            if (out.size() > (4u << 20)) out.erase(0, out.size() - (2u << 20));  // keep the tail of huge output
        }
    });
    const bool timed_out = WaitForSingleObject(pi.hProcess, (DWORD) timeout_s * 1000) == WAIT_TIMEOUT;
    if (timed_out) TerminateJobObject(job, 1);
    WaitForSingleObject(pi.hProcess, 5000);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(job);  // also ends anything it left running in the background
    CloseHandle(pi.hProcess);
    rd.join();
    CloseHandle(out_r);
    out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
    return { { "exit_code", (int) code }, { "timed_out", timed_out }, { "output", cut(out, 12000) } };
}

std::string pick_folder() {
    std::string picked;
    std::thread t([&] {  // the dialog needs its own single-threaded COM apartment
        if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) return;
        IFileOpenDialog * dlg = nullptr;
        if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dlg)))) {
            DWORD opts = 0;
            dlg->GetOptions(&opts);
            dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);
            dlg->SetTitle(L"Zorlo: choose the project folder");
            HWND owner = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"STATIC", L"", WS_POPUP, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
            if (SUCCEEDED(dlg->Show(owner))) {
                IShellItem * item = nullptr;
                if (SUCCEEDED(dlg->GetResult(&item))) {
                    PWSTR path = nullptr;
                    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) { picked = narrow(path); CoTaskMemFree(path); }
                    item->Release();
                }
            }
            if (owner) DestroyWindow(owner);
            dlg->Release();
        }
        CoUninitialize();
    });
    t.join();
    return picked;
}
#else
json run_command(const std::string &, int) { throw std::runtime_error("commands need Windows for now"); }
std::string pick_folder() { return ""; }
#endif

json web_search(const std::string & q) {
    json results = json::array();
    std::string body, err;
    const std::string key = G.settings.value("brave_api_key", "");
    if (!key.empty()) {
        if (!zorlo_http_get("https://api.search.brave.com/res/v1/web/search?count=8&q=" + url_encode(q),
                            "Accept: application/json\r\nX-Subscription-Token: " + key + "\r\n", body, err, 2 << 20))
            throw std::runtime_error("Brave Search: " + err);
        for (const auto & r : json::parse(body).value("web", json::object()).value("results", json::array()))
            results.push_back({ { "title", r.value("title", "") }, { "url", r.value("url", "") },
                                { "snippet", html_to_text(r.value("description", ""), 400) } });
        return { { "engine", "brave" }, { "results", results } };
    }
    if (!zorlo_http_get("https://lite.duckduckgo.com/lite/?q=" + url_encode(q), "Accept: text/html\r\n", body, err, 2 << 20))
        throw std::runtime_error("search failed: " + err);
    std::regex link("<a[^>]*?href=\"([^\"]+)\"[^>]*class=['\"]result-link['\"][^>]*>([\\s\\S]*?)</a>");
    std::regex snip("<td[^>]*class=['\"]result-snippet['\"][^>]*>([\\s\\S]*?)</td>");
    std::vector<std::string> snippets;
    for (std::sregex_iterator it(body.begin(), body.end(), snip), end; it != end; ++it) snippets.push_back(html_to_text((*it)[1], 400));
    size_t i = 0;
    for (std::sregex_iterator it(body.begin(), body.end(), link), end; it != end && results.size() < 8; ++it, ++i) {
        std::string url = decode_entities((*it)[1]);
        if (const size_t p = url.find("uddg="); p != std::string::npos) url = url_decode(url.substr(p + 5, url.find('&', p) - p - 5));
        if (url.rfind("//", 0) == 0) url = "https:" + url;
        if (url.find("duckduckgo.com/y.js") != std::string::npos) continue;  // ads
        results.push_back({ { "title", html_to_text((*it)[2], 200) }, { "url", url }, { "snippet", i < snippets.size() ? snippets[i] : "" } });
    }
    if (results.empty())
        return { { "engine", "duckduckgo" }, { "results", results },
                 { "note", "no results (the free search may be busy; a Brave Search key in Settings makes search reliable)" } };
    return { { "engine", "duckduckgo" }, { "results", results } };
}

json fetch_url(const std::string & url) {
    std::string body, err;
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) throw std::runtime_error("the address must start with http:// or https://");
    if (!zorlo_http_get(url, "Accept: text/html,application/json,text/plain,*/*\r\n", body, err, 3 << 20)) throw std::runtime_error(err);
    const bool html = body.find("<html") != std::string::npos || body.find("<HTML") != std::string::npos || body.find("<!DOCTYPE") != std::string::npos ||
                      body.find("<!doctype") != std::string::npos;
    return { { "url", url }, { "text", html ? html_to_text(body, 20000) : cut(body, 20000) } };
}

std::string preview_address(const fs::path & p) {
    std::string rel = p.lexically_relative(fs::weakly_canonical(workspace())).generic_u8string(), enc;
    for (char c : rel) enc += c == ' ' ? std::string("%20") : std::string(1, c);
    return "http://127.0.0.1:" + std::to_string(G.port) + "/preview/" + enc;
}

// ---------- other model providers (OpenAI-compatible): the agent with the user's own API key ----------
#ifdef _WIN32
// one HTTP(S) request; the answer is handed over piece by piece as it arrives (streamed replies). Returns the status.
int http_request(const std::string & method, const std::string & url, const std::string & headers, const std::string & body,
                 const std::function<bool(const char *, size_t)> & on_data, std::string & err, int * status_now = nullptr) {
    std::wstring wurl = widen(url);
    URL_COMPONENTS uc = { sizeof uc };
    wchar_t host[256], path[4096];
    uc.lpszHostName = host; uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path; uc.dwUrlPathLength = 4096;
    wchar_t extra[4096];
    uc.lpszExtraInfo = extra; uc.dwExtraInfoLength = 4096;
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) { err = "not a valid address: " + url; return 0; }
    const bool tls = uc.nScheme == INTERNET_SCHEME_HTTPS;
    HINTERNET s = WinHttpOpen(L"Zorlo/0.1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s) s = WinHttpOpen(L"Zorlo/0.1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s) { err = "cannot open a connection"; return 0; }
    WinHttpSetTimeouts(s, 15000, 15000, 30000, 600000);  // a slow model may take minutes to answer
    int status = 0;
    HINTERNET c = WinHttpConnect(s, host, uc.nPort, 0), r = nullptr;
    if (c) r = WinHttpOpenRequest(c, widen(method).c_str(), (std::wstring(path) + extra).c_str(), nullptr, WINHTTP_NO_REFERER,
                                  WINHTTP_DEFAULT_ACCEPT_TYPES, tls ? WINHTTP_FLAG_SECURE : 0);
    if (!r) err = "cannot connect to " + narrow(host);
    else {
        const std::wstring h = widen(headers);
        if (!WinHttpSendRequest(r, h.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : h.c_str(), (DWORD) -1, (LPVOID) (body.empty() ? nullptr : body.data()),
                                (DWORD) body.size(), (DWORD) body.size(), 0) || !WinHttpReceiveResponse(r, nullptr)) {
            const DWORD e = GetLastError();
            err = e == ERROR_WINHTTP_NAME_NOT_RESOLVED ? "cannot find " + narrow(host) + " (offline, or a wrong address?)"
                : e == ERROR_WINHTTP_CANNOT_CONNECT ? "cannot connect to " + narrow(host) + (tls ? "" : " (is that program running?)")
                : e == ERROR_WINHTTP_TIMEOUT ? "no answer from " + narrow(host) + " in time" : "connection to " + narrow(host) + " failed (" + std::to_string(e) + ")";
        } else {
            DWORD code = 0, sz = sizeof code;
            WinHttpQueryHeaders(r, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &code, &sz, WINHTTP_NO_HEADER_INDEX);
            status = (int) code;
            if (status_now) *status_now = status;  // known before the first piece of the answer
            char buf[16384];
            for (DWORD n = 0; WinHttpReadData(r, buf, sizeof buf, &n) && n;)
                if (!on_data(buf, n)) break;
        }
    }
    if (r) WinHttpCloseHandle(r);
    if (c) WinHttpCloseHandle(c);
    WinHttpCloseHandle(s);
    return status;
}
#else
int http_request(const std::string &, const std::string &, const std::string &, const std::string &,
                 const std::function<bool(const char *, size_t)> &, std::string & err, int * = nullptr) { err = "only on Windows for now"; return 0; }
#endif

struct Provider { std::string base, key, model; };
Provider provider_locked() {
    const json p = G.settings.value("provider", json::object());
    Provider r;
    r.base = p.value("base_url", "");
    while (!r.base.empty() && r.base.back() == '/') r.base.pop_back();
    r.model = p.value("model", "");
    const json keys = G.settings.value("provider_keys", json::object());
    if (keys.contains(r.base) && keys[r.base].is_string()) r.key = keys[r.base].get<std::string>();
    return r;
}
std::string provider_headers(const Provider & p) {
    std::string h = "Content-Type: application/json\r\nAccept: text/event-stream, application/json\r\n";
    if (!p.key.empty()) h += "Authorization: Bearer " + p.key + "\r\n";
    if (p.base.find("openrouter.ai") != std::string::npos) h += "X-Title: Zorlo\r\n";  // the app name OpenRouter shows
    return h;
}
// a readable message from a provider's error answer ({"error": {"message": ...}} in most APIs)
std::string provider_error(int status, const std::string & body) {
    std::string msg;
    try {
        const json j = json::parse(body);
        if (j.contains("error")) msg = j["error"].is_object() ? j["error"].value("message", j["error"].dump()) : j["error"].dump();
        else if (j.contains("message")) msg = j.value("message", "");
    } catch (...) { msg = cut(body, 300); }
    const std::string hint = status == 401 || status == 403 ? " Check the API key in Settings → Model."
                           : status == 404 ? " Check the model name and the address in Settings → Model."
                           : status == 429 ? " The provider is limiting requests or the account has no credit left." : "";
    while (!msg.empty() && (isspace((unsigned char) msg.back()) || msg.back() == '.')) msg.pop_back();
    return "the provider answered " + std::to_string(status) + ": " + (msg.empty() ? "no details" : msg) + "." + hint;
}

// ---------- files the model can't read directly: documents, PDFs, pictures, video ----------
bool looks_binary(const std::string & head);  // below
std::string file_kind(const fs::path & p) {
    std::string e = p.extension().u8string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    static const std::map<std::string, std::string> kinds = {
        { ".docx", "docx" }, { ".docm", "docx" }, { ".pptx", "pptx" }, { ".pptm", "pptx" }, { ".xlsx", "xlsx" }, { ".xlsm", "xlsx" },
        { ".odt", "odt" }, { ".ods", "ods" }, { ".odp", "odp" }, { ".pdf", "pdf" },
        { ".png", "image" }, { ".jpg", "image" }, { ".jpeg", "image" }, { ".gif", "image" }, { ".bmp", "image" }, { ".tif", "image" },
        { ".tiff", "image" }, { ".webp", "image" }, { ".heic", "image" }, { ".ico", "image" },
        { ".mp4", "media" }, { ".mov", "media" }, { ".mkv", "media" }, { ".avi", "media" }, { ".wmv", "media" }, { ".webm", "media" },
        { ".m4v", "media" }, { ".mp3", "media" }, { ".wav", "media" }, { ".m4a", "media" }, { ".flac", "media" }, { ".ogg", "media" },
        { ".doc", "old" }, { ".xls", "old" }, { ".ppt", "old" } };
    const auto f = kinds.find(e);
    return f == kinds.end() ? "" : f->second;
}

json extract_file(const fs::path & p) {
    std::error_code ec;
    if (!fs::is_regular_file(p, ec)) throw std::runtime_error("no such file: " + p.u8string());
    const std::string kind = file_kind(p);
    if (kind.empty()) {  // plain text, code, CSV, JSON, ...
        std::string t = read_text(p, 400000);
        if (looks_binary(t.substr(0, 8000))) throw std::runtime_error("this is a binary file Zorlo can't turn into text");
        return { { "kind", "text" }, { "text", t } };
    }
    if (kind == "old")
        throw std::runtime_error("old Office format (" + p.extension().u8string() + "): open it in Word / Excel / PowerPoint and save it as .docx / .xlsx / .pptx");
#ifdef _WIN32
    const fs::path dir = appdata_dir() / "tmp";
    fs::create_directories(dir, ec);
    const fs::path script = dir / "extract.ps1", out = dir / (random_token().substr(0, 12) + ".json");
    { std::ofstream s(script, std::ios::binary); s << "param($Path, $Kind, $Out)\n" << EXTRACT_PS; }
    auto q = [](std::string s) { for (size_t i = 0; (i = s.find('\'', i)) != std::string::npos; i += 2) s.insert(i, "'"); return "'" + s + "'"; };
    const json run = run_command("& " + q(script.u8string()) + " -Path " + q(p.u8string()) + " -Kind " + q(kind) + " -Out " + q(out.u8string()), 600);
    std::string text;
    try { text = read_text(out); } catch (...) {
        throw std::runtime_error("could not read the file: " + cut(run.value("output", ""), 600));
    }
    fs::remove(out, ec);
    if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
    json r = json::parse(text);
    if (r.contains("error") && r["error"].is_string()) throw std::runtime_error(r["error"].get<std::string>());
    return r;
#else
    throw std::runtime_error("reading documents needs Windows for now");
#endif
}

// what the user is asked to approve, in one line
std::string summary_of(const std::string & name, const json & a) {
    auto s = [&](const char * k) { return a.contains(k) && a[k].is_string() ? a[k].get<std::string>() : std::string(); };
    if (name == "run_command") return s("command");
    if (name == "write_file") return "write " + s("path") + " (" + std::to_string(s("content").size()) + " characters)";
    if (name == "edit_file") return "edit " + s("path");
    if (name == "read_file" || name == "list_dir" || name == "read_document") return s("path");
    if (name == "web_search") return s("query");
    if (name == "fetch_url") return s("url");
    if (name == "add_connector") {  // never show the keys themselves
        std::string t = "add connector " + (!s("id").empty() ? s("id") : !s("url").empty() ? s("url") : s("command") + " " +
                        (a.contains("args") ? a["args"].dump() : ""));
        if (a.contains("values") && a["values"].is_object() && !a["values"].empty()) {
            t += " with ";
            for (const auto & [k, v] : a["values"].items()) t += k + " ";
        }
        return t;
    }
    if (name == "remove_connector") return "remove connector " + s("name");
    if (name == "use_connector") return s("connector") + " · " + s("tool") + " " + (a.contains("arguments") ? a["arguments"].dump().substr(0, 260) : "");
    if (name == "connector_tools") return s("connector") + (s("tool").empty() ? "" : " · " + s("tool"));
    if (name == "install_skill") return "install skill " + (!s("url").empty() ? s("url") : s("name"));
    if (name == "find_connectors" || name == "find_skills") return s("query");
    return a.dump().substr(0, 300);
}

// folders a coding agent should not wander into
bool skip_dir(const fs::path & p) {
    static const char * skip[] = { ".git", "node_modules", "__pycache__", ".venv", "venv", ".zorlo", ".next", ".cache", ".idea", ".vs" };
    const std::string n = p.filename().u8string();
    return std::any_of(std::begin(skip), std::end(skip), [&](const char * s) { return n == s; });
}

bool glob_match(const std::string & pat, const std::string & s) {  // * and ?, case-insensitive
    size_t p = 0, i = 0, star = std::string::npos, mark = 0;
    auto eq = [](char a, char b) { return tolower((unsigned char) a) == tolower((unsigned char) b); };
    while (i < s.size()) {
        if (p < pat.size() && (pat[p] == '?' || eq(pat[p], s[i]))) { ++p; ++i; }
        else if (p < pat.size() && pat[p] == '*') { star = p++; mark = i; }
        else if (star != std::string::npos) { p = star + 1; i = ++mark; }
        else return false;
    }
    while (p < pat.size() && pat[p] == '*') ++p;
    return p == pat.size();
}

bool looks_binary(const std::string & head) { return head.find('\0') != std::string::npos; }

// every file under root (skipping the folders above), as paths relative to root
std::vector<fs::path> walk_files(const fs::path & root, size_t max) {
    std::vector<fs::path> out;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator() && out.size() < max; it.increment(ec)) {
        if (ec) break;
        if (it->is_directory(ec)) { if (skip_dir(it->path())) it.disable_recursion_pending(); continue; }
        if (it->is_regular_file(ec)) out.push_back(it->path().lexically_relative(root));
    }
    return out;
}

// undo: before the agent changes a file, keep what was there (or that nothing was) under %APPDATA%\Zorlo\backups
fs::path backup_dir() { return appdata_dir() / "backups"; }
std::string make_backup(const fs::path & p) {
    std::error_code ec;
    fs::create_directories(backup_dir(), ec);
    const std::string id = random_token().substr(0, 16);
    const bool existed = fs::is_regular_file(p, ec);
    if (existed) fs::copy_file(p, backup_dir() / (id + ".bak"), fs::copy_options::overwrite_existing, ec);
    std::ofstream(backup_dir() / (id + ".json"), std::ios::binary) << json{ { "path", p.u8string() }, { "existed", existed } }.dump();
    return id;
}

json run_tool(const std::string & name, const json & a) {
    auto str = [&](const char * k) {
        if (!a.contains(k) || !a[k].is_string()) throw std::runtime_error(std::string("missing \"") + k + "\"");
        return a[k].get<std::string>();
    };
    bool inside = true;
    if (name == "list_dir") {
        const fs::path p = resolve(a.value("path", "."), inside);
        const int depth = std::clamp(a.value("depth", 1), 1, 4);
        json items = json::array();
        size_t n = 0;
        std::function<void(const fs::path &, int, const std::string &)> walk = [&](const fs::path & d, int lvl, const std::string & indent) {
            std::vector<fs::directory_entry> es;
            for (const auto & e : fs::directory_iterator(d)) es.push_back(e);
            std::sort(es.begin(), es.end(), [](const auto & x, const auto & y) {
                return x.is_directory() != y.is_directory() ? x.is_directory() : x.path().filename() < y.path().filename(); });
            for (const auto & e : es) {
                if (++n > 400) { if (n == 401) items.push_back("... more entries not shown"); return; }
                std::error_code ec;
                const std::string nm = indent + e.path().filename().u8string();
                if (e.is_directory()) {
                    items.push_back(nm + "/" + (skip_dir(e.path()) ? "  (not listed)" : ""));
                    if (lvl < depth && !skip_dir(e.path())) walk(e.path(), lvl + 1, indent + "  ");
                } else items.push_back(nm + "  (" + std::to_string(e.file_size(ec)) + " bytes)");
            }
        };
        walk(p, 1, "");
        return { { "path", p.u8string() }, { "entries", items } };
    }
    if (name == "search_files") {
        const fs::path root = resolve(a.value("path", "."), inside);
        const std::string q = str("query"), glob = a.value("glob", "");
        const bool use_re = a.value("regex", false);
        std::regex re;
        if (use_re) re = std::regex(q, std::regex::icase | std::regex::ECMAScript);
        std::string ql = q;
        std::transform(ql.begin(), ql.end(), ql.begin(), ::tolower);
        json hits = json::array();
        size_t files = 0;
        for (const auto & rel : walk_files(root, 5000)) {
            if (!glob.empty() && !glob_match(glob, rel.filename().u8string()) && !glob_match(glob, rel.generic_u8string())) continue;
            std::error_code ec;
            if (fs::file_size(root / rel, ec) > (1u << 20)) continue;
            std::string text;
            try { text = read_text(root / rel); } catch (...) { continue; }
            if (looks_binary(text.substr(0, 8000))) continue;
            ++files;
            std::istringstream in(text);
            std::string line;
            for (int no = 1; std::getline(in, line); ++no) {
                bool hit;
                if (use_re) hit = std::regex_search(line, re);
                else { std::string l = line; std::transform(l.begin(), l.end(), l.begin(), ::tolower); hit = l.find(ql) != std::string::npos; }
                if (!hit) continue;
                if (line.size() > 200) line = line.substr(0, 200) + "...";
                hits.push_back(rel.generic_u8string() + ":" + std::to_string(no) + ": " + line);
                if (hits.size() >= 150) return { { "matches", hits }, { "note", "stopped at 150 matches; search more narrowly" } };
            }
        }
        return { { "matches", hits }, { "files_searched", files } };
    }
    if (name == "read_file") {
        const fs::path p = resolve(str("path"), inside);
        const std::string text = read_text(p, 4 << 20);
        const int off = std::max(1, a.value("offset", 1)), lim = std::clamp(a.value("limit", 400), 1, 2000);
        std::istringstream in(text);
        std::string line, out;
        int no = 0, total = 0;
        while (std::getline(in, line)) {
            ++total;
            if (total >= off && no < lim) { out += std::to_string(total) + "\t" + line + "\n"; ++no; }
        }
        json r = { { "path", p.u8string() }, { "content", cut(out, 24000) }, { "total_lines", total } };
        if (off + no - 1 < total) r["note"] = "more lines after " + std::to_string(off + no - 1) + "; read again with offset";
        return r;
    }
    if (name == "write_file") {
        const fs::path p = resolve(str("path"), inside);
        fs::create_directories(p.parent_path());
        const std::string c = str("content");
        std::error_code ec;
        const bool existed = fs::exists(p, ec);
        const std::string undo = make_backup(p);
        { std::ofstream o(p, std::ios::binary); o << c; if (!o) throw std::runtime_error("cannot write " + p.u8string()); }
        json r = { { "path", p.u8string() }, { "bytes", c.size() }, { "created", !existed }, { "undo_id", undo } };
        const std::string ext = p.extension().u8string();
        if (inside && (ext == ".html" || ext == ".htm")) r["preview"] = preview_address(p);
        return r;
    }
    if (name == "edit_file") {
        const fs::path p = resolve(str("path"), inside);
        std::string text = read_text(p);
        const std::string o = str("old_text"), n = str("new_text");
        if (o.empty()) throw std::runtime_error("old_text is empty");
        const size_t first = text.find(o);
        if (first == std::string::npos) throw std::runtime_error("old_text was not found in the file; read the file again and copy the text exactly");
        if (text.find(o, first + 1) != std::string::npos) throw std::runtime_error("old_text appears more than once; include more surrounding lines");
        text.replace(first, o.size(), n);
        const std::string undo = make_backup(p);
        { std::ofstream f(p, std::ios::binary); f << text; if (!f) throw std::runtime_error("cannot write " + p.u8string()); }
        const int line = (int) std::count(text.begin(), text.begin() + (long long) first, '\n') + 1;
        json r = { { "path", p.u8string() }, { "replaced", true }, { "line", line }, { "undo_id", undo } };
        const std::string ext = p.extension().u8string();
        if (inside && (ext == ".html" || ext == ".htm" || ext == ".css" || ext == ".js")) r["changed_web_file"] = true;
        return r;
    }
    if (name == "read_document") {
        const fs::path p = resolve(str("path"), inside);
        json r = extract_file(p);
        std::string t = r.value("text", "");
        const size_t off = (size_t) std::max(0, a.value("offset", 0)), page = 20000;
        r["total_characters"] = t.size();
        if (off >= t.size() && !t.empty()) throw std::runtime_error("offset is past the end (" + std::to_string(t.size()) + " characters)");
        r["text"] = t.substr(std::min(off, t.size()), page);
        if (off + page < t.size()) r["more"] = "read again with offset " + std::to_string(off + page);
        r["path"] = p.u8string();
        return r;
    }
    if (name == "run_command") return run_command(str("command"), std::clamp(a.value("timeout_s", 120), 5, 1800));
    if (name == "web_search") return web_search(str("query"));
    if (name == "fetch_url") return fetch_url(str("url"));
    if (name == "preview_url") {
        const fs::path p = resolve(str("path"), inside);
        if (!inside) throw std::runtime_error("only files inside the project folder can be previewed");
        if (!fs::exists(p)) throw std::runtime_error("no such file: " + p.u8string());
        return { { "url", preview_address(p) } };
    }
    if (name == "load_skill") {
        const std::string want = str("name");
        for (const auto & s : list_skills())
            if (s.name == want || s.file.parent_path().filename().u8string() == want) {
                std::string t = read_text(s.file, 60000);
                if (t.rfind("---", 0) == 0) if (const size_t e = t.find("\n---", 3); e != std::string::npos) t = t.substr(t.find('\n', e + 1) + 1);
                return { { "skill", s.name }, { "folder", s.file.parent_path().u8string() }, { "instructions", t } };
            }
        throw std::runtime_error("no skill named " + want);
    }
    if (name == "find_connectors") {
        const std::string q = a.value("query", "");
        json installed = json::object(), out = json::array();
        { std::lock_guard<std::mutex> lk(G.mu); for (const auto & [n, c] : G.settings["mcpServers"].items()) installed[c.value("market", n)] = n; }
        for (const auto & e : market().value("connectors", json::array())) {
            if (!matches(e, q)) continue;
            json needs = json::array();
            for (const auto & f : e.value("fields", json::array())) needs.push_back({ { "key", f.value("key", "") }, { "label", f.value("label", "") }, { "kind", f.value("kind", "") } });
            json o = { { "id", e.value("id", "") }, { "name", e.value("name", "") }, { "about", e.value("about", "") }, { "needs", needs },
                       { "runtime", e.value("runtime", "") } };
            if (installed.contains(o["id"].get<std::string>())) o["installed_as"] = installed[o["id"].get<std::string>()];
            out.push_back(o);
        }
        return { { "connectors", out }, { "installed_programs", runtimes() },
                 { "note", out.empty() ? "nothing matched; any MCP server can still be added with command/args or url" : "" } };
    }
    if (name == "add_connector") return add_connector(a);
    if (name == "remove_connector") return remove_connector(str("name"));
    if (name == "find_skills") {
        json out = json::array(), have = json::array();
        for (const auto & s : list_skills()) have.push_back(s.file.parent_path().filename().u8string());
        for (const auto & e : skill_market()) {
            if (!matches(e, a.value("query", ""))) continue;
            json o = e;
            o["installed"] = std::find(have.begin(), have.end(), e.value("folder", "")) != have.end();
            out.push_back(o);
        }
        return { { "skills", out }, { "note", "any other skill on GitHub can be installed by its link" } };
    }
    if (name == "install_skill") return install_skill(a);
    if (name == "connector_tools") {
        const std::string c = str("connector"), only = a.value("tool", "");
        std::lock_guard<std::mutex> lk(G.mu);
        const auto f = G.mcp.find(c);
        if (f == G.mcp.end() || !f->second->error.empty()) {
            std::string names;
            for (const auto & [n, s] : G.mcp) if (s->error.empty()) names += (names.empty() ? "" : ", ") + n;
            throw std::runtime_error("no running connector named " + c + (names.empty() ? "" : "; running: " + names));
        }
        json out = json::array();
        for (const auto & tt : f->second->tools) {
            if (!only.empty() && tt.value("name", "") != only) continue;
            out.push_back({ { "tool", tt.value("name", "") }, { "description", cut(tt.value("description", ""), 1200) },
                            { "inputSchema", tt.value("inputSchema", json::object()) } });
        }
        if (out.empty()) throw std::runtime_error(c + " has no tool named " + only);
        return { { "connector", c }, { "tools", out } };
    }
    if (name.rfind("mcp__", 0) == 0 || name == "use_connector") {
        std::pair<std::string, std::string> st;
        McpServer * srv = nullptr;
        json args = a;
        {
            std::lock_guard<std::mutex> lk(G.mu);
            if (name == "use_connector") {
                st = { str("connector"), str("tool") };
                const auto f = G.mcp.find(st.first);
                if (f == G.mcp.end() || !f->second->error.empty()) throw std::runtime_error("no running connector named " + st.first);
                const auto & tl = f->second->tools;
                if (std::none_of(tl.begin(), tl.end(), [&](const json & x) { return x.value("name", "") == st.second; }))
                    throw std::runtime_error(st.first + " has no tool named " + st.second + "; see connector_tools");
                args = a.value("arguments", json::object());
                if (args.is_string()) args = json::parse(args.get<std::string>());
                srv = f->second.get();
            } else {
                if (!G.mcp_tool.count(name)) throw std::runtime_error("unknown connector tool " + name);
                st = G.mcp_tool[name];
                srv = G.mcp[st.first].get();
            }
        }
        // a failed call shows the arguments the tool takes, so the model can fix the call at once
        auto schema = [&] {
            std::lock_guard<std::mutex> lk(G.mu);
            const auto f = G.mcp.find(st.first);
            if (f != G.mcp.end())
                for (const auto & tt : f->second->tools) if (tt.value("name", "") == st.second) return tt.value("inputSchema", json::object());
            return json::object();
        };
        json res;
        try { res = srv->request("tools/call", { { "name", st.second }, { "arguments", args } }, 300); }
        catch (const std::exception & e) {
            return { { "is_error", true }, { "result", std::string("error: ") + e.what() }, { "arguments_it_takes", schema() } };
        }
        std::string text;
        for (const auto & c : res.value("content", json::array()))
            text += c.value("type", "") == "text" ? c.value("text", "") + "\n" : "[" + c.value("type", "") + " content]\n";
        json r = { { "result", cut(text, 20000) } };
        if (res.value("isError", false)) { r["is_error"] = true; r["arguments_it_takes"] = schema(); }
        return r;
    }
    throw std::runtime_error("unknown tool " + name);
}

// ---------- HTTP plumbing ----------
bool allowed(const httplib::Request & req, httplib::Response & res, bool need_token) {
    const std::string host = req.get_header_value("Host"), port = ":" + std::to_string(G.port);
    const bool host_ok = host == "127.0.0.1" + port || host == "localhost" + port;
    const std::string origin = req.get_header_value("Origin");
    const bool origin_ok = origin.empty() || origin == "http://127.0.0.1" + port || origin == "http://localhost" + port;
    const bool token_ok = !need_token || req.get_header_value("X-Zorlo-Token") == G.token;
    if (G.tools_on && host_ok && origin_ok && token_ok) return true;
    res.status = 403;
    res.set_content(json{ { "error", G.tools_on ? "not allowed" : "agent tools are only available when Zorlo serves on 127.0.0.1" } }.dump(),
                    "application/json");
    return false;
}

const char * mime_of(const std::string & ext) {
    static const std::map<std::string, const char *> m = {
        { ".html", "text/html; charset=utf-8" }, { ".htm", "text/html; charset=utf-8" }, { ".css", "text/css; charset=utf-8" },
        { ".js", "text/javascript; charset=utf-8" }, { ".mjs", "text/javascript; charset=utf-8" }, { ".json", "application/json" },
        { ".svg", "image/svg+xml" }, { ".png", "image/png" }, { ".jpg", "image/jpeg" }, { ".jpeg", "image/jpeg" },
        { ".gif", "image/gif" }, { ".webp", "image/webp" }, { ".ico", "image/x-icon" }, { ".txt", "text/plain; charset=utf-8" },
        { ".md", "text/plain; charset=utf-8" }, { ".woff2", "font/woff2" }, { ".woff", "font/woff" }, { ".mp4", "video/mp4" },
        { ".wasm", "application/wasm" } };
    std::string e = ext;
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    const auto it = m.find(e);
    return it == m.end() ? "application/octet-stream" : it->second;
}

// this launch's token into a page (only when the agent tools are on: loopback)
std::string with_token(std::string html) {
    const std::string t = G.tools_on ? G.token : "";
    for (size_t p; (p = html.find("{{ZORLO_TOKEN}}")) != std::string::npos;) html.replace(p, 15, t);
    return html;
}

void serve_file(const fs::path & root, const std::string & rel_url, httplib::Response & res) {
    std::error_code ec;
    const fs::path base = fs::weakly_canonical(root, ec);
    fs::path p = fs::weakly_canonical(base / fs::u8path(rel_url), ec);  // httplib already decoded the request path
    const auto rel = p.lexically_relative(base);
    if (ec || rel.empty() || *rel.begin() == "..") { res.status = 404; return; }
    if (fs::is_directory(p, ec)) p /= "index.html";
    if (!fs::is_regular_file(p, ec)) { res.status = 404; res.set_content("not found", "text/plain"); return; }
    res.set_content(read_text(p), mime_of(p.extension().u8string()));
    res.set_header("Cache-Control", "no-cache");
}

}  // namespace

void zorlo_agent_register(httplib::Server & svr, const std::string & host, int port, const char * fallback_html) {
    G.host = host;
    G.port = port;
    G.tools_on = host == "127.0.0.1" || host == "localhost";
    G.token = random_token();
    G.settings_path = fs::u8path(env_or("ZORLO_SETTINGS", (appdata_dir() / "settings.json").u8string()));
    G.ui_dir = fs::u8path(env_or("ZORLO_UI_DIR", (exe_dir() / "ui").u8string()));
    load_settings();
    const std::string fallback = fallback_html;
    // is there a newer Zorlo? the model catalog says (read once, in the background, so nothing waits for it)
    std::thread([] {
        std::string text, err;
        const std::string src = env_or("ZORLO_CATALOG", "https://huggingface.co/datasets/Haidershahi1/zorlo-catalog/resolve/main/models.json");
        if (src.rfind("https://", 0) == 0) { if (!zorlo_fetch_text(src, text, err)) return; }
        else try { text = read_text(fs::u8path(src)); } catch (...) { return; }  // a local catalog (testing)
        try {
            if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
            const json z = json::parse(text).value("zorlo", json::object());
            const std::string latest = z.value("latest", "");
            int newer = 0;  // version compare, part by part
            for (std::istringstream a(ZORLO_VERSION), b(latest);;) {
                std::string pa, pb;
                const bool ha = (bool) std::getline(a, pa, '.'), hb = (bool) std::getline(b, pb, '.');
                if (!ha && !hb) break;
                const long va = ha ? atol(pa.c_str()) : 0, vb = hb ? atol(pb.c_str()) : 0;
                if (va != vb) { newer = vb > va ? 1 : -1; break; }
            }
            if (newer > 0) { std::lock_guard<std::mutex> lk(G.mu); G.update = { { "latest", latest }, { "url", z.value("url", "") }, { "notes", z.value("notes", "") } }; }
        } catch (...) {}
    }).detach();

    // the app: ui\index.html with this launch's token; the simple built-in chat when the ui folder is missing
    svr.Get("/", [fallback](const httplib::Request &, httplib::Response & res) {
        std::error_code ec;
        const fs::path index = G.ui_dir / "index.html";
        if (G.tools_on && fs::exists(index, ec)) {
            std::string html = read_text(index);
            for (size_t p; (p = html.find("{{ZORLO_TOKEN}}")) != std::string::npos;) html.replace(p, 15, G.token);
            res.set_content(html, "text/html; charset=utf-8");
            res.set_header("Cache-Control", "no-store");
        } else res.set_content(with_token(fallback), "text/html; charset=utf-8");
    });
    svr.Get("/simple", [fallback](const httplib::Request &, httplib::Response & res) {
        res.set_content(with_token(fallback), "text/html; charset=utf-8");
        res.set_header("Cache-Control", "no-store");
    });
    svr.Get(R"(/ui/(.+))", [](const httplib::Request & req, httplib::Response & res) { serve_file(G.ui_dir, req.matches[1], res); });
    svr.Get(R"(/preview/(.*))", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, false)) return;
        std::lock_guard<std::mutex> lk(G.mu);
        serve_file(workspace(), req.matches[1], res);
    });

    svr.Get("/agent/state", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            std::lock_guard<std::mutex> lk(G.mu);
            json skills = json::array(), conns = json::array();
            for (const auto & s : list_skills()) skills.push_back({ { "name", s.name }, { "description", s.description } });
            const json tools = tool_list_locked();
            const json servers = G.settings.value("mcpServers", json::object());  // named: items() must not outlive it
            for (const auto & [n, cfg] : servers.items()) {
                const auto f = G.mcp.find(n);
                const bool off = cfg.value("disabled", false);
                std::string cmd = cfg.value("command", "");
                for (const auto & x : cfg.value("args", json::array())) if (x.is_string()) cmd += " " + x.get<std::string>();
                conns.push_back({ { "name", n }, { "market", cfg.value("market", "") }, { "enabled", !off }, { "command", cmd },
                                  { "starting", !off && f == G.mcp.end() && G.mcp_starting },
                                  { "ok", !off && f != G.mcp.end() && f->second->error.empty() },
                                  { "error", off ? "" : f == G.mcp.end() ? "not started" : f->second->error },
                                  { "tools", f == G.mcp.end() ? 0 : f->second->tools.size() },
                                  { "tool_names", [&] { json n = json::array(); if (!off && f != G.mcp.end() && f->second->error.empty())
                                      for (const auto & t : f->second->tools) n.push_back(t.value("name", "")); return n; }() } });
            }
            // project instructions, like CLAUDE.md / AGENTS.md: the first of these in the project folder goes into the prompt
            json instructions = nullptr;
            for (const char * f : { "ZORLO.md", "AGENTS.md", "CLAUDE.md" }) {
                std::error_code ec;
                if (fs::is_regular_file(workspace() / f, ec)) { instructions = { { "file", f }, { "text", read_text(workspace() / f, 8000) } }; break; }
            }
            json shown = G.settings;  // API keys stay in the server: the page only learns whether one is set
            shown.erase("provider_keys");
            shown["brave_key_set"] = !shown.value("brave_api_key", "").empty();
            shown["brave_api_key"] = "";
            shown["provider"]["has_key"] = !provider_locked().key.empty();
            res.set_content(json{ { "version", ZORLO_VERSION }, { "update", G.update }, { "settings", shown }, { "settings_file", G.settings_path.u8string() },
                                  { "tools", tools }, { "skills", skills }, { "connectors", conns }, { "instructions", instructions } }
                                .dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        } catch (const std::exception & e) {
            res.status = 500;
            res.set_content(json{ { "error", e.what() } }.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        }
    });
    svr.Post("/agent/settings", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            const json in = json::parse(req.body);
            std::lock_guard<std::mutex> lk(G.mu);
            const json old_mcp = G.settings.value("mcpServers", json::object()), old_skills = G.settings.value("skills_dir", "");
            const std::string brave_before = G.settings.value("brave_api_key", "");
            for (const auto & [k, v] : in.items()) if (default_settings().contains(k) && k != "provider_keys") G.settings[k] = v;
            // the page never has the saved key: an empty field keeps it, unless the user cleared it on purpose
            if (in.contains("brave_api_key") && in.value("brave_api_key", "").empty() && !in.value("brave_key_clear", false))
                G.settings["brave_api_key"] = brave_before;
            if (in.contains("provider")) {
                json & p = G.settings["provider"];
                p.erase("has_key");
                std::string base = p.value("base_url", "");
                while (!base.empty() && (base.back() == '/' || base.back() == ' ')) base.pop_back();
                base.erase(0, base.find_first_not_of(' '));
                p["base_url"] = base;
                if (p.value("kind", "zorlo") != "zorlo" && !base.empty() && base.rfind("http://", 0) != 0 && base.rfind("https://", 0) != 0)
                    throw std::runtime_error("the provider address must start with https:// (or http:// for a program on this PC)");
                // a key belongs to one address: switching provider never sends it somewhere else
                if (in.contains("provider_key") && in["provider_key"].is_string() && !in["provider_key"].get<std::string>().empty() && !base.empty())
                    G.settings["provider_keys"][base] = in["provider_key"].get<std::string>();
                if (in.value("provider_key_clear", false)) G.settings["provider_keys"].erase(base);
            }
            for (const auto & [g, v] : G.settings["permissions"].items())
                if (v != "allow" && v != "ask" && v != "deny") G.settings["permissions"][g] = "ask";
            std::error_code ec;
            fs::create_directories(workspace(), ec);
            if (!fs::is_directory(workspace(), ec)) throw std::runtime_error("the project folder doesn't exist and can't be created");
            save_settings();
            if (G.settings["mcpServers"] != old_mcp) G.mcp_started = false;  // restarted on the next tool list
            res.set_content(json{ { "ok", true } }.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", e.what() } }.dump(), "application/json");
        }
    });
    // undo one change the agent made (write_file / edit_file return its undo_id)
    svr.Post("/agent/undo", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            const std::string id = json::parse(req.body).at("undo_id").get<std::string>();
            if (id.size() != 16 || !std::all_of(id.begin(), id.end(), ::isxdigit)) throw std::runtime_error("bad undo id");
            const json meta = json::parse(read_text(backup_dir() / (id + ".json")));
            const fs::path p = fs::u8path(meta.at("path").get<std::string>());
            if (meta.value("existed", false)) fs::copy_file(backup_dir() / (id + ".bak"), p, fs::copy_options::overwrite_existing);
            else fs::remove(p);
            res.set_content(json{ { "ok", true }, { "path", p.u8string() }, { "restored", meta.value("existed", false) } }.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", e.what() } }.dump(), "application/json");
        }
    });
    // the user attaches a file: saved into the project (default folder "uploads"); body = the raw bytes
    svr.Post("/agent/upload", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            std::string name = req.get_param_value("name"), dir = req.has_param("dir") ? req.get_param_value("dir") : "uploads";
            for (char & c : name) if (strchr("\\/:*?\"<>|", c) || (unsigned char) c < 32) c = '_';
            if (name.empty() || name == "." || name == "..") name = "file";
            bool inside = true;
            fs::path d;
            { std::lock_guard<std::mutex> lk(G.mu); d = resolve(dir, inside); }
            if (!inside) throw std::runtime_error("uploads go inside the project folder");
            fs::create_directories(d);
            fs::path p = d / fs::u8path(name);
            for (int i = 2; fs::exists(p); ++i)
                p = d / fs::u8path(fs::u8path(name).stem().u8string() + " (" + std::to_string(i) + ")" + fs::u8path(name).extension().u8string());
            { std::ofstream o(p, std::ios::binary); o << req.body; if (!o) throw std::runtime_error("cannot write " + p.u8string()); }
            const std::string rel = p.lexically_relative(fs::weakly_canonical(workspace())).generic_u8string();
            res.set_content(json{ { "path", rel }, { "bytes", req.body.size() } }.dump(), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", e.what() } }.dump(), "application/json");
        }
    });
    // ---- chats, kept in %APPDATA%\Zorlo\chats\<kind>\<id>.json: the same list in Zorlo Desktop and any browser ----
    auto chat_dir = [](const httplib::Request & req) {
        const std::string kind = req.has_param("kind") && req.get_param_value("kind") == "simple" ? "simple" : "agent";
        return appdata_dir() / "chats" / kind;
    };
    auto safe_chat_id = [](const std::string & id) {
        return !id.empty() && id.size() <= 40 && std::all_of(id.begin(), id.end(), [](char c) { return isalnum((unsigned char) c) || c == '-' || c == '_'; });
    };
    svr.Get("/agent/chats", [chat_dir](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        std::vector<std::pair<double, json>> list;
        std::error_code ec;
        for (const auto & e : fs::directory_iterator(chat_dir(req), ec)) {
            if (e.path().extension() != ".json") continue;
            try { json c = json::parse(read_text(e.path())); list.push_back({ c.value("updated", 0.0), std::move(c) }); } catch (...) {}
        }
        std::sort(list.begin(), list.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        json out = json::array();
        for (size_t i = 0; i < list.size() && i < 300; ++i) out.push_back(std::move(list[i].second));
        res.set_content(json{ { "chats", out } }.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Post("/agent/chat", [chat_dir, safe_chat_id](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            const json c = json::parse(req.body);
            const std::string id = c.value("id", "");
            if (!safe_chat_id(id)) throw std::runtime_error("bad chat id");
            if (req.body.size() > (64u << 20)) throw std::runtime_error("chat too large");
            const fs::path dir = chat_dir(req);
            fs::create_directories(dir);
            const fs::path f = dir / (id + ".json"), tmp = dir / (id + ".json.tmp");
            { std::ofstream o(tmp, std::ios::binary); o << req.body; if (!o) throw std::runtime_error("cannot write the chat"); }
            fs::rename(tmp, f);
            res.set_content("{\"ok\":true}", "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", e.what() } }.dump(), "application/json");
        }
    });
    svr.Post("/agent/chat_delete", [chat_dir, safe_chat_id](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            const std::string id = json::parse(req.body).value("id", "");
            if (!safe_chat_id(id)) throw std::runtime_error("bad chat id");
            std::error_code ec;
            fs::remove(chat_dir(req) / (id + ".json"), ec);
            res.set_content("{\"ok\":true}", "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", e.what() } }.dump(), "application/json");
        }
    });

    // turn an attached file into text for the model: ?path= a project file, or ?name= with the file's bytes as the body
    svr.Post("/agent/extract", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        std::error_code ec;
        fs::path tmp;
        try {
            fs::path p;
            if (req.has_param("path")) {
                bool inside = true;
                { std::lock_guard<std::mutex> lk(G.mu); p = resolve(req.get_param_value("path"), inside); }
                if (!inside) throw std::runtime_error("outside the project folder");
            } else {
                std::string name = req.get_param_value("name");
                for (char & c : name) if (strchr("\\/:*?\"<>|", c) || (unsigned char) c < 32) c = '_';
                const fs::path dir = appdata_dir() / "tmp" / random_token().substr(0, 12);
                fs::create_directories(dir);
                tmp = dir;
                p = dir / fs::u8path(name.empty() ? "file" : name);
                std::ofstream(p, std::ios::binary) << req.body;
            }
            json r = extract_file(p);
            r["name"] = p.filename().u8string();
            res.set_content(r.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", e.what() } }.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        }
        if (!tmp.empty()) fs::remove_all(tmp, ec);
    });
    // project files for the Files panel and @ mentions
    svr.Get("/agent/files", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        fs::path ws;
        { std::lock_guard<std::mutex> lk(G.mu); ws = workspace(); }
        json files = json::array();
        std::error_code ec;
        for (const auto & rel : walk_files(ws, 3000)) files.push_back({ rel.generic_u8string(), fs::file_size(ws / rel, ec) });
        res.set_content(json{ { "root", ws.u8string() }, { "files", files } }.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Get("/agent/file", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            bool inside = true;
            fs::path p;
            { std::lock_guard<std::mutex> lk(G.mu); p = resolve(req.get_param_value("path"), inside); }
            if (!inside) throw std::runtime_error("outside the project folder");
            std::string t = read_text(p, 600000);
            if (looks_binary(t.substr(0, 8000))) throw std::runtime_error("binary file");
            res.set_content(json{ { "path", req.get_param_value("path") }, { "content", t } }.dump(-1, ' ', false, json::error_handler_t::replace),
                            "application/json");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", e.what() } }.dump(), "application/json");
        }
    });
    svr.Post("/agent/pick_folder", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        const std::string p = pick_folder();
        res.set_content(p.empty() ? json{ { "cancelled", true } }.dump() : json{ { "path", p } }.dump(), "application/json");
    });
    // ---- another model provider: the page's chat requests go out with the user's key (the page never sees it) ----
    svr.Post("/agent/llm", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        Provider p;
        json body;
        try {
            body = json::parse(req.body);
            { std::lock_guard<std::mutex> lk(G.mu); p = provider_locked(); }
            if (p.base.rfind("http://", 0) != 0 && p.base.rfind("https://", 0) != 0) throw std::runtime_error("set the provider's address in Settings → Model");
            if (p.model.empty()) throw std::runtime_error("set the model name in Settings → Model");
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{ { "error", { { "message", e.what() } } } }.dump(), "application/json");
            return;
        }
        body["model"] = p.model;
        body.erase("mode");  // Zorlo's own options: other servers reject fields they don't know
        body.erase("chat_template_kwargs");
        const bool stream = body.value("stream", false);
        if (stream) body["stream_options"] = { { "include_usage", true } };
        const std::string payload = body.dump(-1, ' ', false, json::error_handler_t::replace);
        res.set_chunked_content_provider(stream ? "text/event-stream" : "application/json", [p, payload, stream](size_t, httplib::DataSink & sink) {
            std::string err, errbody;
            int now = 0;
            const int status = http_request("POST", p.base + "/chat/completions", provider_headers(p), payload, [&](const char * d, size_t n) {
                if (now == 200) return sink.write(d, n);  // the reply, passed on as it streams in
                if (errbody.size() < 65536) errbody.append(d, n);  // an error: kept to explain it below
                return true;
            }, err, &now);
            if (status != 200) {  // a message the page can show
                const std::string msg = status == 0 ? err : provider_error(status, errbody);
                const std::string line = stream ? "\n\ndata: " + json{ { "error", { { "message", msg } } } }.dump() + "\n\n"
                                                : json{ { "error", { { "message", msg } } } }.dump();
                sink.write(line.data(), line.size());
            }
            sink.done();
            return true;
        });
    });
    // the provider's model list, for the model field in Settings
    svr.Get("/agent/llm/models", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        Provider p;
        { std::lock_guard<std::mutex> lk(G.mu); p = provider_locked(); }
        std::string body, err;
        const int status = http_request("GET", p.base + "/models", provider_headers(p), "", [&](const char * d, size_t n) {
            if (body.size() < (8u << 20)) body.append(d, n);
            return true;
        }, err);
        json out = { { "models", json::array() } };
        if (status == 200) {
            try {
                const json j = json::parse(body);
                for (const auto & m : j.value("data", j.value("models", json::array())))
                    if (m.is_object()) out["models"].push_back(m.value("id", m.value("name", "")));
            } catch (...) { out["error"] = "the provider's model list could not be read"; }
        } else out["error"] = status == 0 ? err : provider_error(status, body);
        res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // ---- connector and skill markets ----
    auto fail = [](httplib::Response & res, const std::exception & e) {
        res.status = 400;
        res.set_content(json{ { "error", e.what() } }.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    };
    svr.Get("/agent/market", [fail](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            json m = market();
            res.set_content(json{ { "connectors", m.value("connectors", json::array()) }, { "runtimes", runtimes() } }.dump(), "application/json");
        } catch (const std::exception & e) { fail(res, e); }
    });
    svr.Get("/agent/skill_market", [fail](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try { res.set_content(json{ { "skills", skill_market() } }.dump(-1, ' ', false, json::error_handler_t::replace), "application/json"); }
        catch (const std::exception & e) { fail(res, e); }
    });
    // {action: add (id+values | url | command+args+env, name), remove, enable, disable}
    svr.Post("/agent/connector", [fail](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            const json in = json::parse(req.body);
            const std::string act = in.value("action", "");
            json r;
            if (act == "add") r = add_connector(in);
            else if (act == "remove") r = remove_connector(in.at("name").get<std::string>());
            else if (act == "enable" || act == "disable") r = set_connector_enabled(in.at("name").get<std::string>(), act == "enable");
            else throw std::runtime_error("unknown action");
            res.set_content(r.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        } catch (const std::exception & e) { fail(res, e); }
    });
    svr.Post("/agent/skill", [fail](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        try {
            const json in = json::parse(req.body);
            const std::string act = in.value("action", "");
            json r;
            if (act == "install") r = install_skill(in);
            else if (act == "remove") r = remove_skill(in.at("name").get<std::string>());
            else throw std::runtime_error("unknown action");
            res.set_content(r.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
        } catch (const std::exception & e) { fail(res, e); }
    });

    svr.Post("/agent/tool", [](const httplib::Request & req, httplib::Response & res) {
        if (!allowed(req, res, true)) return;
        json out;
        struct FolderScope { ~FolderScope() { t_folder.clear(); } } folder_scope;
        try {
            const json in = json::parse(req.body);
            if (in.contains("folder") && in["folder"].is_string() && !in["folder"].get<std::string>().empty()) {
                std::error_code ec;
                const std::string f = in["folder"].get<std::string>();
                if (!fs::is_directory(fs::u8path(f), ec)) throw std::runtime_error("this chat's project folder no longer exists: " + f);
                t_folder = f;
            }
            const std::string name = in.at("name").get<std::string>();
            json args = in.value("arguments", json::object());
            if (args.is_string()) args = json::parse(args.get<std::string>());  // OpenAI tool calls carry a JSON string
            const bool approved = in.value("approved", false);
            std::string perm, group = group_of(name);
            bool outside = false;
            {
                std::lock_guard<std::mutex> lk(G.mu);
                if (group.empty()) throw std::runtime_error("unknown tool " + name);
                const bool full = G.settings.value("full_permission", false);
                perm = group == "skills" ? "allow" : G.settings["permissions"].value(group, "ask");
                if (full && perm != "deny") perm = "allow";
                if (args.contains("path") && args["path"].is_string()) {
                    bool inside = true;
                    resolve(args["path"].get<std::string>(), inside);
                    outside = !inside;
                    if (outside && !full && perm == "allow") perm = "ask";  // outside the project folder: always ask
                }
            }
            out["summary"] = summary_of(name, args);
            out["group"] = group;
            if (outside) out["outside_project"] = true;
            if (perm == "deny") {
                out["status"] = "blocked";
                out["result"] = "This tool is blocked in Zorlo's settings (" + group + "). Tell the user what you wanted to do instead.";
            } else if (perm == "ask" && !approved) {
                out["status"] = "needs_approval";
            } else {
                out["status"] = "ok";
                out["result"] = run_tool(name, args);
            }
        } catch (const std::exception & e) {
            out["status"] = "error";
            out["result"] = std::string("error: ") + e.what();
        }
        res.set_content(out.dump(), "application/json");
    });
}
