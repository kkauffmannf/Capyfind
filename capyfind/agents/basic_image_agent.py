"""
Basic Image Generator Agent for Capyfind

This is a simple LangChain agent that:
1. Uses Groq LLM to create creative prompts
2. Calls HuggingFace API to generate images
3. Demonstrates basic agent patterns

Perfect for learning how AI agents work!
"""

import os
import requests
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

# Load environment variables
load_dotenv()

class ImageResult(BaseModel):
    """Simple model for image generation results"""
    success: bool
    image_path: Optional[str] = None
    prompt_used: Optional[str] = None
    error_message: Optional[str] = None

class BasicImageAgent:
    """
    A simple image generation agent using LangChain + Groq + HuggingFace
    
    This agent demonstrates:
    - LLM integration with Groq
    - API calls to HuggingFace
    - Basic error handling
    - Structured outputs
    """
    
    def __init__(self):
        """Initialize the agent with LLM and API keys"""
        
        # Set up Groq LLM for creative prompts
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.8  # High creativity for image prompts
        )
        
        # Get HuggingFace API key
        self.hf_api_key = os.getenv("HF_API_KEY")
        
        print("🎨 BasicImageAgent initialized!")
        print(f"   - LLM Model: {self.llm.model_name}")
        print(f"   - API Keys: {'✅' if self.hf_api_key else '❌'}")
    
    def create_prompt(self, theme: str = "forest") -> str:
        """
        Use the LLM to create a detailed image generation prompt
        """
        
        system_message = """You are a creative prompt engineer. Create a detailed prompt for generating a complex background image suitable for a hidden object game.

The prompt should describe:
- A visually rich and complex scene
- Lots of details, patterns, and textures
- Good hiding spots for small objects
- Avoid mentioning people or animals

Keep it under 200 words and focus on visual richness."""

        user_message = f"Create an image generation prompt for a {theme} scene"
        
        # Call the LLM
        messages = [HumanMessage(content=f"{system_message}\n\n{user_message}")]
        response = self.llm.invoke(messages)
        
        prompt = response.content.strip()
        print(f"🧠 LLM created prompt: {prompt[:80]}...")
        
        return prompt
    
    def generate_image(self, prompt: str) -> ImageResult:
        """
        Generate an image using HuggingFace API
        """
        
        try:
            # HuggingFace API endpoint
            url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            
            headers = {
                "Authorization": f"Bearer {self.hf_api_key}"
            }
            
            # API request data
            data = {
                "inputs": prompt,
                "parameters": {
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 7.5,
                    "negative_prompt": "people, animals, text, watermark, low quality"
                }
            }
            
            print(f"🖼️ Calling HuggingFace API...")
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}.png"
                filepath = f"data/{filename}"
                
                # Make sure data directory exists
                os.makedirs("data", exist_ok=True)
                
                # Write image file
                with open(filepath, "wb") as f:
                    f.write(response.content)
                
                print(f"✅ Image saved: {filepath}")
                
                return ImageResult(
                    success=True,
                    image_path=filepath,
                    prompt_used=prompt
                )
                
            else:
                error_msg = f"HuggingFace API error: {response.status_code}"
                print(f"❌ {error_msg}")
                
                return ImageResult(
                    success=False,
                    error_message=error_msg
                )
                
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            
            return ImageResult(
                success=False,
                error_message=error_msg
            )
    
    def create_scene(self, theme: str = "forest") -> ImageResult:
        """
        Main agent method: Create a complete scene
        
        This combines the LLM and image generation steps
        """
        
        print(f"\n🎯 Agent creating {theme} scene...")
        
        # Step 1: Create prompt with LLM
        prompt = self.create_prompt(theme)
        
        # Step 2: Generate image
        result = self.generate_image(prompt)
        
        if result.success:
            print(f"🎉 Agent completed successfully!")
        else:
            print(f"😞 Agent failed: {result.error_message}")
        
        return result

# Simple function for easy use
def generate_scene(theme: str = "forest") -> ImageResult:
    """
    Easy-to-use function for generating scenes
    """
    agent = BasicImageAgent()
    return agent.create_scene(theme)