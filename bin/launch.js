#!/usr/bin/env node

import { execSync, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const SERVER = join(ROOT, "server.py");
const REQUIREMENTS = join(ROOT, "requirements.txt");
const VENV = join(ROOT, ".venv");

// Find a working Python >= 3.10
function findPython() {
  for (const cmd of ["python3", "python"]) {
    try {
      const version = execSync(`${cmd} --version 2>&1`, {
        encoding: "utf-8",
      }).trim();
      const match = version.match(/Python (\d+)\.(\d+)/);
      if (match && (Number(match[1]) > 3 || (Number(match[1]) === 3 && Number(match[2]) >= 10))) {
        return cmd;
      }
    } catch {
      // not found, try next
    }
  }
  return null;
}

const python = findPython();
if (!python) {
  console.error("Error: Python 3.10+ is required but not found on PATH.");
  console.error("Install it from https://www.python.org/downloads/");
  process.exit(1);
}

// Create venv and install deps if needed
const venvPython = process.platform === "win32"
  ? join(VENV, "Scripts", "python.exe")
  : join(VENV, "bin", "python");

if (!existsSync(venvPython)) {
  console.error("Setting up virtual environment...");
  execSync(`${python} -m venv "${VENV}"`, { cwd: ROOT, stdio: "inherit" });
  console.error("Installing dependencies...");
  execSync(`"${venvPython}" -m pip install -r "${REQUIREMENTS}" --quiet`, {
    cwd: ROOT,
    stdio: "inherit",
  });
}

// Launch the MCP server with stdio transport
const server = spawn(venvPython, [SERVER], {
  cwd: ROOT,
  stdio: ["inherit", "inherit", "inherit"],
  env: { ...process.env, MCP_TRANSPORT: "stdio" },
});

server.on("exit", (code) => process.exit(code ?? 1));
process.on("SIGINT", () => server.kill("SIGINT"));
process.on("SIGTERM", () => server.kill("SIGTERM"));
