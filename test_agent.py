#!/usr/bin/env python3
"""
Test script for the Basic Image Generator Agent

This script demonstrates how to use your first AI agent!
"""

import sys
import os

# Add project to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_agent():
    """Test our basic image generation agent"""
    
    print("🦫 Testing Basic Image Generator Agent")
    print("=" * 50)
    
    try:
        # Import our agent
        from capyfind.agents.basic_image_agent import BasicImageAgent, generate_scene
        
        print("✅ Agent imported successfully!")
        
        # Test 1: Create an agent instance
        print("\n🔧 Creating agent instance...")
        agent = BasicImageAgent()
        
        # Test 2: Generate a prompt
        print("\n🧠 Testing prompt creation...")
        prompt = agent.create_prompt("mystical forest")
        print(f"   Generated prompt length: {len(prompt)} characters")
        print(f"   Preview: {prompt[:100]}...")
        
        # Test 3: Generate an image
        print("\n🖼️ Testing full scene generation...")
        result = agent.create_scene("enchanted garden")
        
        print(f"\n📊 Results:")
        print(f"   Success: {result.success}")
        
        if result.success:
            print(f"   Image saved: {result.image_path}")
            print(f"   Prompt used: {result.prompt_used[:80]}...")
            
            # Check if file actually exists
            if os.path.exists(result.image_path):
                print(f"   ✅ File verified: {os.path.getsize(result.image_path)} bytes")
            else:
                print(f"   ⚠️ File not found at expected path")
        else:
            print(f"   Error: {result.error_message}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_simple_function():
    """Test the simple generate_scene function"""
    
    print(f"\n📦 Testing Simple Function Interface")
    print("-" * 30)
    
    try:
        from capyfind.agents.basic_image_agent import generate_scene
        
        print("🎯 Using generate_scene() function...")
        result = generate_scene("bustling marketplace")
        
        if result.success:
            print(f"✅ Simple function works!")
            print(f"   Image: {result.image_path}")
            return True
        else:
            print(f"❌ Simple function failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Simple function test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Basic Agent Tests")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_basic_agent()
    test2_passed = test_simple_function()
    
    print(f"\n📋 Test Results")
    print("=" * 30)
    print(f"Basic Agent Test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Simple Function Test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 Congratulations! Your first AI agent is working!")
        print(f"📁 Check the 'data/' folder for generated images")
    else:
        print(f"\n⚠️ Some tests failed. Check your API keys in .env file")
        print(f"💡 Make sure you have valid GROQ_API_KEY and HF_API_KEY")