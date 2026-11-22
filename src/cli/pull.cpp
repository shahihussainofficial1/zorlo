// zorlo pull <owner/name>[@revision] [out_dir] [--lowbit]
//
// Downloads a Zorlo pack from a Hugging Face model repo: every file in the repo root except the optional
// low-bit copy (experts.lo.bin, ~9.5 GB, only with --lowbit). Downloads go to <file>.part and resume
// after a cut connection or Ctrl+C; each file is checked against the SHA-256 (LFS) or git SHA-1 the
// Hugging Face API lists before it is renamed into place. HF_TOKEN is sent for private repos, only to
// huggingface.co (redirects to the download CDN carry their own signed URL).
// Pulling again updates the pack: <out_dir>/.zorlo-pull.json records each file's checksum, and a file whose
// checksum changed on the Hub is downloaded again (the old copy stays until the new one is verified).
//
// Windows only (WinHTTP + BCrypt, both part of Windows).

#include "cli/pull.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#ifndef _WIN32
int zorlo_pull_main(int, char **) {
    fprintf(stderr, "zorlo pull: only on Windows for now\n");
    return 1;
}
int zorlo_pull(std::string, std::string, bool, bool, bool) { return zorlo_pull_main(0, nullptr); }
bool zorlo_fetch_text(const std::string &, std::string &, std::string & err) { err = "only on Windows for now"; return false; }
bool zorlo_http_get(const std::string &, const std::string &, std::string &, std::string & err, size_t) { err = "only on Windows for now"; return false; }
std::string zorlo_models_dir() { return "models"; }
void zorlo_pull_json(bool) {}
const std::string & zorlo_pull_last_error() { static std::string e = "only on Windows for now"; return e; }
#else

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#include <bcrypt.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using clk = std::chrono::steady_clock;

static std::wstring widen(const std::string & s) {
    std::wstring w(MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), (int) w.size());
    return w;
}

static std::string narrow(const std::wstring & w) {
    std::string s(WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), (int) s.size(), nullptr, nullptr);
    return s;
}

static std::string pct(const std::string & s) {  // one URL path segment
    static const char * hex = "0123456789ABCDEF";
    std::string o;
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char) c;
        else { o += '%'; o += hex[c >> 4]; o += hex[c & 15]; }
    }
    return o;
}

static std::string human(double bytes) {
    char b[32];
    snprintf(b, sizeof b, bytes >= 1e9 ? "%.2f GB" : "%.1f MB", bytes >= 1e9 ? bytes / 1e9 : bytes / 1e6);
    return b;
}

struct Handle {
    HINTERNET h = nullptr;
    Handle() = default;
    explicit Handle(HINTERNET x) : h(x) {}
    Handle(Handle && o) noexcept : h(o.h) { o.h = nullptr; }
    Handle & operator=(Handle && o) noexcept { std::swap(h, o.h); return *this; }
    ~Handle() { if (h) WinHttpCloseHandle(h); }
};

struct Response {
    Handle conn, req;
    DWORD status = 0;
};

class Http {
  public:
    Http() {
        session_.h = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Zorlo/0.1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                                 WINHTTP_NO_PROXY_BYPASS, 0);
        if (!session_.h)  // WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY needs Windows 8.1+
            session_.h = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) Zorlo/0.1", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0);
        if (!session_.h) throw std::runtime_error("WinHttpOpen failed");
        DWORD tls = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | 0x2000 /* TLS 1.3 */;
        if (!WinHttpSetOption(session_.h, WINHTTP_OPTION_SECURE_PROTOCOLS, &tls, sizeof tls)) {
            tls = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;  // older Windows without TLS 1.3
            WinHttpSetOption(session_.h, WINHTTP_OPTION_SECURE_PROTOCOLS, &tls, sizeof tls);
        }
        WinHttpSetTimeouts(session_.h, 0, 30000, 30000, 60000);
        if (const char * t = getenv("HF_TOKEN")) token_ = t;
        else if (const char * t2 = getenv("HUGGING_FACE_HUB_TOKEN")) token_ = t2;
    }

    // GET with redirects followed by hand (so the token never leaves huggingface.co); 200/206 or throws.
    // extra_hdr: more request headers ("Name: value\r\n" each); allow_http: plain http too (web pages for the agent)
    Response get(std::string url, uint64_t from = 0, const std::wstring & extra_hdr = L"", bool allow_http = false) {
        for (int hop = 0; hop < 10; ++hop) {
            const std::wstring wurl = widen(url);
            URL_COMPONENTS u = { sizeof u };
            wchar_t host[256], path[8192];
            u.lpszHostName = host;
            u.dwHostNameLength = 256;
            u.lpszUrlPath = path;
            u.dwUrlPathLength = 8192;
            wchar_t extra[8192];
            u.lpszExtraInfo = extra;
            u.dwExtraInfoLength = 8192;
            if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &u)) throw std::runtime_error("bad URL " + url);
            const bool https = u.nScheme == INTERNET_SCHEME_HTTPS;
            if (!https && !(allow_http && u.nScheme == INTERNET_SCHEME_HTTP)) throw std::runtime_error("refusing non-HTTPS URL " + url);
            const std::wstring whost(host, u.dwHostNameLength);
            const std::wstring target = std::wstring(path, u.dwUrlPathLength) + std::wstring(extra, u.dwExtraInfoLength);

            Response r;
            r.conn.h = WinHttpConnect(session_.h, whost.c_str(), u.nPort, 0);
            if (!r.conn.h) throw std::runtime_error("cannot connect to " + narrow(whost) + err());
            r.req.h = WinHttpOpenRequest(r.conn.h, L"GET", target.c_str(), nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES, https ? WINHTTP_FLAG_SECURE : 0);
            if (!r.req.h) throw std::runtime_error("cannot open request" + err());
            DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
            WinHttpSetOption(r.req.h, WINHTTP_OPTION_REDIRECT_POLICY, &policy, sizeof policy);

            std::wstring hdr;
            const std::string h = narrow(whost);
            const bool hf = h == "huggingface.co" || (h.size() > 15 && h.compare(h.size() - 15, 15, ".huggingface.co") == 0);
            if (hf && !token_.empty()) hdr += L"Authorization: Bearer " + widen(token_) + L"\r\n";
            if (from) hdr += L"Range: bytes=" + std::to_wstring(from) + L"-\r\n";
            hdr += extra_hdr;
            if (!WinHttpSendRequest(r.req.h, hdr.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : hdr.c_str(),
                                    hdr.empty() ? 0 : (DWORD) -1L, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
                !WinHttpReceiveResponse(r.req.h, nullptr))
                throw std::runtime_error("request to " + h + " failed" + err());
            DWORD sz = sizeof r.status;
            WinHttpQueryHeaders(r.req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                                &r.status, &sz, WINHTTP_NO_HEADER_INDEX);
            if (r.status == 301 || r.status == 302 || r.status == 303 || r.status == 307 || r.status == 308) {
                std::wstring loc(4096, L'\0');
                DWORD lb = (DWORD) (loc.size() * sizeof(wchar_t));
                if (!WinHttpQueryHeaders(r.req.h, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, loc.data(), &lb,
                                         WINHTTP_NO_HEADER_INDEX))
                    throw std::runtime_error("redirect without Location");
                loc.resize(lb / sizeof(wchar_t));
                std::string next = narrow(loc);
                if (next.rfind("/", 0) == 0)
                    next = std::string(https ? "https://" : "http://") + h +
                           (u.nPort == (https ? 443 : 80) ? "" : ":" + std::to_string(u.nPort)) + next;
                url = next;
                continue;
            }
            if (r.status == 200 || r.status == 206) return r;
            std::string body = read_all(r);
            if (body.size() > 300) body.resize(300);
            if (hf && (r.status == 401 || r.status == 403))
                throw std::runtime_error("HTTP " + std::to_string(r.status) +
                                         ": no such repo, or it is private/gated (then set HF_TOKEN to a Hugging Face"
                                         " access token)");
            if (r.status == 404) throw std::runtime_error("HTTP 404, not found: " + url);
            throw std::runtime_error("HTTP " + std::to_string(r.status) + " from " + h + ": " + body);
        }
        throw std::runtime_error("too many redirects");
    }

    static std::string read_all(Response & r) {
        std::string out;
        char buf[65536];
        DWORD n = 0;
        while (WinHttpReadData(r.req.h, buf, sizeof buf, &n) && n) out.append(buf, n);
        return out;
    }

  private:
    static std::string err() { return " (error " + std::to_string(GetLastError()) + ")"; }
    Handle session_;
    std::string token_;
};

class Hasher {
  public:
    explicit Hasher(const wchar_t * alg) {
        if (BCryptOpenAlgorithmProvider(&alg_, alg, nullptr, 0) || BCryptCreateHash(alg_, &h_, nullptr, 0, nullptr, 0, 0))
            throw std::runtime_error("BCrypt hash init failed");
    }
    ~Hasher() {
        if (h_) BCryptDestroyHash(h_);
        if (alg_) BCryptCloseAlgorithmProvider(alg_, 0);
    }
    void update(const void * p, size_t n) { BCryptHashData(h_, (PUCHAR) p, (ULONG) n, 0); }
    std::string hex() {
        DWORD len = 0, got = 0;
        BCryptGetProperty(alg_, BCRYPT_HASH_LENGTH, (PUCHAR) &len, sizeof len, &got, 0);
        std::vector<unsigned char> d(len);
        BCryptFinishHash(h_, d.data(), len, 0);
        std::string s;
        char b[3];
        for (unsigned char c : d) { snprintf(b, sizeof b, "%02x", c); s += b; }
        return s;
    }
  private:
    BCRYPT_ALG_HANDLE alg_ = nullptr;
    BCRYPT_HASH_HANDLE h_ = nullptr;
};

struct RemoteFile {
    std::string name, sha256, sha1;  // sha256 for LFS files, git blob sha1 otherwise
    uint64_t size = 0;
};

static bool g_json = false;                       // zorlo start --gui: progress as JSON lines on stdout
static std::string g_last_error;
static uint64_t g_all_need = 0, g_all_done = 0;  // bytes this pull downloads, and what finished files added

void zorlo_pull_json(bool on) { g_json = on; }
const std::string & zorlo_pull_last_error() { return g_last_error; }

std::string zorlo_models_dir() {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof buf / sizeof buf[0]));
    return (fs::path(std::wstring(buf, n)).parent_path() / "models").u8string();
}

struct Progress {
    clk::time_point t0 = clk::now(), last{};
    uint64_t start = 0;
    void show(const std::string & name, uint64_t done, uint64_t total, bool force = false) {
        const auto now = clk::now();
        if (!force && now - last < std::chrono::milliseconds(g_json ? 300 : 500)) return;
        last = now;
        const double s = std::chrono::duration<double>(now - t0).count();
        const double rate = s > 0 ? (done - start) / s : 0;
        if (g_json) {
            printf("%s\n", json{ { "type", "progress" }, { "file", name }, { "done", done }, { "total", total }, { "rate", (uint64_t) rate },
                                 { "all_done", g_all_done + (done - start) }, { "all_total", g_all_need } }.dump().c_str());
            fflush(stdout);
            return;
        }
        char eta[32] = "";
        if (rate > 0 && done < total) {
            const int left = (int) ((total - done) / rate);
            snprintf(eta, sizeof eta, "  %dm%02ds left", left / 60, left % 60);
        }
        fprintf(stderr, "\r  %-22s %s / %s  %.1f MB/s%s      ", name.c_str(), human((double) done).c_str(),
                human((double) total).c_str(), rate / 1e6, eta);
    }
};

// <dir>/.zorlo-pull.json remembers which version (checksum) of each file is on disk, so a later pull can tell
// "up to date" from "changed on the Hub" without re-reading gigabytes
static const char * MARKER = ".zorlo-pull.json";
static json load_marker(const fs::path & dir) {
    std::ifstream in(dir / MARKER);
    if (!in) return json::object();
    try { return json::parse(in); } catch (...) { return json::object(); }
}
static void save_marker(const fs::path & dir, const json & m) {
    const fs::path tmp = dir / (std::string(MARKER) + ".tmp");
    { std::ofstream o(tmp, std::ios::binary); o << m.dump(2); if (!o) throw std::runtime_error("cannot write " + tmp.string()); }
    fs::rename(tmp, dir / MARKER);
}
static std::string want_id(const RemoteFile & f) { return f.sha256.empty() ? f.sha1 : f.sha256; }

// checksum of a local file the way the Hub lists it (SHA-256 for LFS files, git blob SHA-1 otherwise)
static std::string local_id(const fs::path & p, const RemoteFile & f) {
    Hasher h(f.sha256.empty() ? BCRYPT_SHA1_ALGORITHM : BCRYPT_SHA256_ALGORITHM);
    if (f.sha256.empty()) { const std::string hdr = "blob " + std::to_string(f.size); h.update(hdr.c_str(), hdr.size() + 1); }
    FILE * in = _wfopen(p.c_str(), L"rb");
    if (!in) return "";
    std::vector<char> buf(4 << 20);
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), in)) > 0) h.update(buf.data(), n);
    fclose(in);
    return h.hex();
}

// is the local copy the version the Hub has now? Files without a record (older pulls, zorlo pack) are
// checksummed once and then remembered.
static bool up_to_date(const RemoteFile & f, const fs::path & dir, json & marker) {
    const fs::path dst = dir / fs::u8path(f.name);
    if (!fs::exists(dst) || fs::file_size(dst) != f.size) return false;
    const json & files = marker["files"];
    if (files.contains(f.name)) return files[f.name].value("id", "") == want_id(f);
    if (f.size > (64 << 20)) fprintf(stderr, "  %-22s checking the copy already on disk...\n", f.name.c_str());
    if (local_id(dst, f) != want_id(f)) return false;
    marker["files"][f.name] = { { "id", want_id(f) }, { "size", f.size } };
    save_marker(dir, marker);
    return true;
}

// download one file into dir (resuming <name>.part), check its hash, rename into place
static void fetch(Http & http, const std::string & base, const RemoteFile & f, const fs::path & dir, json & marker) {
    const fs::path dst = dir / fs::u8path(f.name), part = dir / fs::u8path(f.name + ".part");
    const bool lfs = !f.sha256.empty();
    Hasher hash(lfs ? BCRYPT_SHA256_ALGORITHM : BCRYPT_SHA1_ALGORITHM);
    if (!lfs) {  // git blob id = sha1("blob <size>\0" + content)
        const std::string hdr = "blob " + std::to_string(f.size);
        hash.update(hdr.c_str(), hdr.size() + 1);
    }

    uint64_t have = 0;
    std::vector<char> buf(4 << 20);
    if (fs::exists(part)) {
        have = fs::file_size(part);
        if (have > f.size) { fs::remove(part); have = 0; }
        if (have) {  // hash what is already there, then continue from its end
            fprintf(stderr, "  %-22s checking %s downloaded earlier...\n", f.name.c_str(), human((double) have).c_str());
            FILE * in = _wfopen(part.c_str(), L"rb");
            if (!in) throw std::runtime_error("cannot read " + part.string());
            uint64_t left = have;
            while (left) {
                const size_t n = fread(buf.data(), 1, (size_t) std::min<uint64_t>(left, buf.size()), in);
                if (!n) break;
                hash.update(buf.data(), n);
                left -= n;
            }
            fclose(in);
            if (left) throw std::runtime_error("short read in " + part.string());
        }
    }

    FILE * out = _wfopen(part.c_str(), L"ab");
    if (!out) throw std::runtime_error("cannot write " + part.string());
    Progress pr;
    pr.start = have;
    const std::string url = base + pct(f.name);
    for (int attempt = 1; have < f.size; ++attempt) {
        try {
            Response r = http.get(url, have);
            if (have && r.status == 200) throw std::runtime_error("server ignored the resume request");
            DWORD n = 0;
            while (have < f.size) {
                if (!WinHttpReadData(r.req.h, buf.data(), (DWORD) buf.size(), &n))
                    throw std::runtime_error("connection lost (error " + std::to_string(GetLastError()) + ")");
                if (!n) break;
                if (have + n > f.size) throw std::runtime_error("server sent more bytes than the file has");
                if (fwrite(buf.data(), 1, n, out) != n) throw std::runtime_error("write failed (disk full?)");
                hash.update(buf.data(), n);
                have += n;
                pr.show(f.name, have, f.size);
            }
            if (have < f.size) throw std::runtime_error("connection closed early");
        } catch (const std::exception & e) {
            if (attempt >= 8) { fclose(out); throw; }
            const int wait = std::min(30, 1 << attempt);
            fprintf(stderr, "\n  %s, retrying in %d s (%d/8)\n", e.what(), wait, attempt);
            std::this_thread::sleep_for(std::chrono::seconds(wait));
        }
    }
    if (fclose(out)) throw std::runtime_error("write failed: " + part.string());
    pr.show(f.name, have, f.size, true);  // before adding it to the total: show() adds this file's own progress
    g_all_done += have - pr.start;
    fprintf(stderr, "\n");

    const std::string got = hash.hex(), want = lfs ? f.sha256 : f.sha1;
    if (got != want) {
        fs::remove(part);
        throw std::runtime_error(f.name + ": checksum mismatch (got " + got + ", expected " + want +
                                 "); the partial file was deleted, run zorlo pull again");
    }
    if (fs::exists(dst)) fs::remove(dst);
    fs::rename(part, dst);
    marker["files"][f.name] = { { "id", want_id(f) }, { "size", f.size } };
    save_marker(dir, marker);
}

// fetch a small text file over HTTPS (the model catalog); false with err set on any failure
bool zorlo_fetch_text(const std::string & url, std::string & out, std::string & err) {
    try {
        Http http;
        Response r = http.get(url);
        out = Http::read_all(r);
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

// GET a web page or API answer for the agent: extra headers ("Name: value\r\n" each), http or https, at most
// max_bytes kept; false with err set on failure
bool zorlo_http_get(const std::string & url, const std::string & headers, std::string & out, std::string & err,
                    size_t max_bytes) {
    try {
        Http http;
        Response r = http.get(url, 0, widen(headers), true);
        out.clear();
        char buf[65536];
        DWORD n = 0;
        while (out.size() < max_bytes && WinHttpReadData(r.req.h, buf, sizeof buf, &n) && n) out.append(buf, n);
        if (out.size() > max_bytes) out.resize(max_bytes);
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

// download or update a pack; 0 = everything on disk matches the Hub
int zorlo_pull(std::string repo, std::string dir, bool lowbit, bool any_repo, bool quiet) {
    std::string rev = "main";
    if (const size_t at = repo.find('@'); at != std::string::npos) { rev = repo.substr(at + 1); repo.resize(at); }
    const size_t slash = repo.find('/');
    if (dir.empty()) {  // the same place zorlo start uses, so a model is never downloaded twice
        std::string name = repo.substr(slash + 1);
        if (name.size() > 6 && name.compare(name.size() - 6, 6, "-zorlo") == 0) name.resize(name.size() - 6);
        dir = (fs::u8path(zorlo_models_dir()) / fs::u8path(name)).u8string();
    }
    g_last_error.clear();
    const std::string repo_url = pct(repo.substr(0, slash)) + "/" + pct(repo.substr(slash + 1));

    try {
        Http http;
        Response r = http.get("https://huggingface.co/api/models/" + repo_url + "/tree/" + pct(rev));
        const json tree = json::parse(Http::read_all(r));

        std::vector<RemoteFile> files;
        bool has_manifest = false;
        for (const auto & e : tree) {
            if (e.value("type", "") != "file") continue;
            RemoteFile f;
            f.name = e.at("path").get<std::string>();
            if (f.name.find('/') != std::string::npos || f.name == ".gitattributes" || f.name == "hot.bin") continue;
            if (f.name.rfind("experts.lo", 0) == 0 && !(lowbit && f.name == "experts.lo.bin")) continue;
            f.size = e.at("size").get<uint64_t>();
            if (e.contains("lfs") && e["lfs"].is_object()) {
                f.sha256 = e["lfs"].at("oid").get<std::string>();
                f.size = e["lfs"].value("size", f.size);
            } else {
                f.sha1 = e.at("oid").get<std::string>();
            }
            has_manifest |= f.name == "manifest.json";
            files.push_back(f);
        }
        if (!has_manifest && !any_repo)
            throw std::runtime_error(repo + " has no manifest.json at its root, so it is not a Zorlo pack");
        if (lowbit && std::none_of(files.begin(), files.end(), [](const RemoteFile & f) { return f.name == "experts.lo.bin"; }))
            fprintf(stderr, "note: %s has no experts.lo.bin, so --lowbit will not work with it\n", repo.c_str());
        std::sort(files.begin(), files.end(), [](const RemoteFile & a, const RemoteFile & b) { return a.size < b.size; });

        const fs::path out = fs::u8path(dir);
        fs::create_directories(out);
        json marker = load_marker(out);
        if (!marker.contains("files") || !marker["files"].is_object()) marker["files"] = json::object();
        marker["repo"] = repo + "@" + rev;
        std::vector<RemoteFile> todo;
        uint64_t total = 0, need = 0;
        for (const auto & f : files) {
            total += f.size;
            if (up_to_date(f, out, marker)) continue;
            const fs::path dst = out / fs::u8path(f.name), part = out / fs::u8path(f.name + ".part");
            if (fs::exists(dst)) fprintf(stderr, "  %-22s a newer version is on the Hub\n", f.name.c_str());
            need += f.size - (fs::exists(part) ? std::min<uint64_t>(fs::file_size(part), f.size) : 0);
            todo.push_back(f);
        }
        if (todo.empty()) {
            fprintf(stderr, "%s is up to date (%s)\n", repo.c_str(), human((double) total).c_str());
            if (!quiet) fprintf(stderr, "Start it with:  zorlo serve \"%s\"\n", out.string().c_str());
            return 0;
        }
        const uint64_t free_b = fs::space(out).available;
        fprintf(stderr, "downloading %s: %zu of %zu files, %s to download (%s free on that drive)\n", repo.c_str(),
                todo.size(), files.size(), human((double) need).c_str(), human((double) free_b).c_str());
        if (need > free_b) throw std::runtime_error("not enough free disk space on that drive (" + human((double) need) + " needed)");

        const std::string base = "https://huggingface.co/" + repo_url + "/resolve/" + pct(rev) + "/";
        g_all_need = need;
        g_all_done = 0;
        if (g_json) { printf("%s\n", json{ { "type", "download" }, { "files", todo.size() }, { "need", need } }.dump().c_str()); fflush(stdout); }
        for (const auto & f : todo) fetch(http, base, f, out, marker);
        fprintf(stderr, quiet ? "download complete\n" : "done. Start it with:  zorlo serve \"%s\"\n", out.string().c_str());
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "\nerror: %s\n", e.what());
        g_last_error = e.what();
        return 1;
    }
}

int zorlo_pull_main(int argc, char ** argv) {
    std::string repo, dir;
    bool lowbit = false, any_repo = false;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--lowbit") lowbit = true;
        else if (s == "--any-repo") any_repo = true;  // testing: allow a repo without manifest.json
        else if (repo.empty()) repo = s;
        else if (dir.empty()) dir = s;
        else { fprintf(stderr, "unexpected argument %s\n", s.c_str()); return 1; }
    }
    if (repo.empty() || repo.find('/') == std::string::npos) {
        fprintf(stderr,
                "usage: zorlo pull <owner/name>[@revision] [out_dir] [--lowbit]\n"
                "  downloads a Zorlo pack from Hugging Face, or updates it when files changed there\n"
                "  (resumes if cut; every file is checksum-checked)\n"
                "  --lowbit also downloads experts.lo.bin (~9.5 GB) for --lowbit / --mode fast\n"
                "  HF_TOKEN=<token> for private or gated repos\n");
        return 1;
    }
    return zorlo_pull(repo, dir, lowbit, any_repo, false);
}
#endif
