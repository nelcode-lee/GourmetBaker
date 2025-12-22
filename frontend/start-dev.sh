#!/bin/bash
# Ensure Node 18 is used
cd "$(dirname "$0")"
nvm use 18 2>/dev/null || nvm use 2>/dev/null || true
npm run dev

