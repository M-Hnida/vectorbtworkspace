#!/usr/bin/env python3
"""
Example script demonstrating how to call the RAG assistant API.
"""

import requests
import json
import argparse

# The base URL for your local API
BASE_URL = "http://localhost:8000"

def query_vectorbt_example(question):
    """
    Sends a question to the /vectorbt/query endpoint.
    """
    print("--- 1. Querying VectorBT Assistant ---")
    url = f"{BASE_URL}/vectorbt/query"
    payload = {"question": question}
    
    print(f"Sending question: '{question}' to {url}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        print("\n[SUCCESS] Server Response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] An error occurred: {e}")
    
    print("\n" + "="*50 + "\n")


def review_code_example(code, question):
    """
    Sends code and a question to the /review/code endpoint.
    """
    print("--- 2. Submitting Code for Review ---")
    url = f"{BASE_URL}/review/code"
    payload = {
        "code": code,
        "question": question
    }
    
    print(f"Sending code snippet for review with question: '{question}'")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        print("\n[SUCCESS] Server Response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] An error occurred: {e}")

    print("\n" + "="*50 + "\n")


def review_file(file_path, question):
    """
    Reads a file and sends its contents for review.
    """
    print(f"--- 3. Reviewing File: {file_path} ---")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            code_content = file.read()
        
        review_code_example(code_content, question)
        
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"\n[ERROR] An error occurred while reading the file: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="API Client for RAG Assistant.")
    parser.add_argument("-q", "--question", type=str, 
                        help="Question for the VectorBT assistant.")
    parser.add_argument("-c", "--code", type=str, 
                        help="Code snippet for review.")
    parser.add_argument("-cq", "--code-question", type=str, 
                        help="Question for code review.")
    parser.add_argument("-f", "--file", type=str,
                        help="Path to a file to review.")
    parser.add_argument("-fq", "--file-question", type=str,
                        help="Question for file review.")

    args = parser.parse_args()

    if args.question:
        query_vectorbt_example(args.question)
    elif args.code and args.code_question:
        review_code_example(args.code, args.code_question)
    elif args.file and args.file_question:
        review_file(args.file, args.file_question)
    else:
        # --- Example 1: Ask a question about VectorBT ---
        vectorbt_question = "How can I plot the drawdown of a portfolio in vectorbt?"
        query_vectorbt_example(vectorbt_question)

        # --- Example 2: Get a review for a snippet of Python code ---
        sample_code = """
import pandas as pd

def calculate_average(prices):
    total = 0
    for price in prices:
        total += price
    return total / len(prices)
"""
        code_review_question = "Is this an efficient way to calculate an average in pandas? How could I improve it?"
        review_code_example(sample_code, code_review_question) 