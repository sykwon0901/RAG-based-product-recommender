# src/amazon_skincare_graphrag/config.py

from __future__ import annotations

import os
from getpass import getpass
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI


def load_env(dotenv_path: str = ".env") -> None:
    """
    Load environment variables from a local .env file.
    """
    load_dotenv(dotenv_path=dotenv_path)


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create an OpenAI client (no hardcoded secrets).
    """
    load_env()
    api_key = api_key or os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API key: ")
    return OpenAI(api_key=api_key)


def get_neo4j_driver(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Create a Neo4j driver (no hardcoded secrets).
    Uses .env first; falls back to a sensible default for local Neo4j.
    """
    load_env()

    uri = uri or os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD") or getpass("Enter Neo4j password: ")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver
