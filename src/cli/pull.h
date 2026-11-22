#pragma once
// Hugging Face downloads (src/cli/pull.cpp)

#include <string>

// `zorlo pull` command line
int zorlo_pull_main(int argc, char ** argv);

// download or update the pack in `repo` (owner/name[@revision]) into `dir`; 0 = on disk and up to date.
// quiet: no "start it with zorlo serve" hint (used by zorlo start)
int zorlo_pull(std::string repo, std::string dir, bool lowbit, bool any_repo, bool quiet);

// fetch a small text file over HTTPS (the model catalog); false with err set on failure
bool zorlo_fetch_text(const std::string & url, std::string & out, std::string & err);

// <folder of zorlo.exe>\models: where zorlo start and zorlo pull (without a folder) keep models
std::string zorlo_models_dir();

// zorlo start --gui: report download progress as JSON lines on stdout instead of a console progress line
void zorlo_pull_json(bool on);

// the reason the last zorlo_pull failed ("" after a success)
const std::string & zorlo_pull_last_error();

// GET for the agent's web tools: http or https, extra headers ("Name: value\r\n" each), at most max_bytes kept
bool zorlo_http_get(const std::string & url, const std::string & headers, std::string & out, std::string & err,
                    size_t max_bytes);
