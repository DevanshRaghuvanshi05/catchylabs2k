#!/usr/bin/env python3
"""
Database Migration & Table Setup Script
----------------------------------------
Initializes PostgreSQL tables (research_runs, sources, evidence, api_calls) if they do not exist.
Enables Row Level Security (RLS) on all tables to lock down Supabase's public REST API.
"""

import sys
import logging
from database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CatchyLabResearcher")

if __name__ == "__main__":
    try:
        logger.info("Executing create_tables.py setup script...")
        init_db()
        logger.info("Table setup completed successfully.")
    except Exception as exc:
        logger.error(f"Error initializing tables: {exc}", exc_info=True)
        sys.exit(1)
