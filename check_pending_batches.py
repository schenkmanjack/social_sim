#!/usr/bin/env python3
"""
Script to check for pending Anthropic batch requests and optionally cancel old ones.
"""

import os
import sys
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, '/home/schenkmanjack/social_sim')

from social_sim.llm_interfaces.llm_interface import AnthropicBackend

def main():
    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Create LLM backend
    llm = AnthropicBackend(api_key=api_key, debug=True)
    
    print("=" * 60)
    print("CHECKING FOR PENDING BATCH REQUESTS")
    print("=" * 60)
    
    # Check for pending batches
    pending_batches = llm.check_pending_batches()
    
    if not pending_batches:
        print("✅ No pending batches found!")
        return
    
    print(f"\n⚠️  Found {len(pending_batches)} pending batches")
    
    # Ask user if they want to cancel old batches
    print("\nOptions:")
    print("1. Cancel batches older than 1 hour")
    print("2. Cancel batches older than 30 minutes") 
    print("3. Cancel ALL pending batches")
    print("4. Just monitor (don't cancel anything)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        cancelled = llm.cancel_pending_batches(max_age_hours=1)
        print(f"Cancelled {cancelled} batches older than 1 hour")
    elif choice == "2":
        cancelled = llm.cancel_pending_batches(max_age_hours=0.5)
        print(f"Cancelled {cancelled} batches older than 30 minutes")
    elif choice == "3":
        cancelled = llm.cancel_pending_batches(max_age_hours=0)
        print(f"Cancelled {cancelled} pending batches")
    elif choice == "4":
        print("Monitoring mode - no batches cancelled")
    else:
        print("Invalid choice, exiting without changes")
    
    # Check again after potential cancellations
    if choice in ["1", "2", "3"]:
        print("\n" + "=" * 40)
        print("CHECKING AGAIN AFTER CANCELLATION")
        print("=" * 40)
        remaining_batches = llm.check_pending_batches()
        if not remaining_batches:
            print("✅ All batches cleared!")
        else:
            print(f"⚠️  {len(remaining_batches)} batches still pending")

if __name__ == "__main__":
    main() 