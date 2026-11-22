#pragma once
// Zorlo agent: app page, tools, settings, MCP connectors and skills (src/cli/agent.cpp)

#include <string>

namespace httplib { class Server; }

// registers "/", "/simple", "/ui/*", "/preview/*" and "/agent/*" on the server; fallback_html is the built-in chat
// page used when the ui folder is missing or the server isn't on 127.0.0.1
void zorlo_agent_register(httplib::Server & svr, const std::string & host, int port, const char * fallback_html);
