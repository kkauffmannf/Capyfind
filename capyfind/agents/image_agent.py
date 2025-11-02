"""
Real LangChain-powered Image Generator Agent for Capyfind game.

This agent uses:
- Groq LLM to generate creative, detailed prompts for hiding spots
- HuggingFace API to create complex background images
- LangChain framework for proper agent architecture
"""

import os
import requests
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field

load_dotenv()

class ImageGenerationResult(BaseModel):
    """Result model for image generation"""
    success: bool = Field(description="Whether image generation succeeded")
    image_path: Optional[str] = Field(default=None, description="Path to generated image file")
    prompt_used: Optional[str] = Field(default=None, description="Final prompt used for generation")
    error_message: Optional[str] = Field(default=None, description="Error message if generation failed")
    generation_time: Optional[str] = Field(default=None, description="Timestamp of generation")

class ImageGeneratorAgent:
    """
    A LangChain-powered agent that generates complex background images for the Capyfind game.
    
    This agent:
    1. Uses Groq LLM to create detailed, creative prompts
    2. Calls HuggingFace API to generate images
    3. Follows proper agent patterns with tools and memory
    """
    
    def __init__(self):
        # Initialize LLM with supported model
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",  # Current supported model for creative tasks
            temperature=0.8  # High creativity for varied prompts
        )
        
        # Agent memory - stores conversation history
        self.memory: list[BaseMessage] = []
        
        # Available capabilities (using modern LangChain patterns)
        self.capabilities = [
            "create_image_prompt",
            "generate_image_with_hf"
        ]
        
        print(f"🎨 ImageGeneratorAgent initialized with Groq LLM and {len(self.capabilities)} capabilities")
    
    def _create_image_prompt(self, scene_type: str = "random") -> str:
        """
        Use Groq LLM to generate a creative, detailed prompt for image generation.
        """
        system_prompt = """You are a creative prompt engineer for a "Where's Waldo?" style hidden object game.

Your task: Create a SINGLE, detailed image generation prompt (not a conversation) for a complex background scene where small capybaras will be hidden later.

Requirements:
- NO animals or people in the scene (capybaras added later)
- Maximum visual complexity: patterns, textures, repetitive elements
- Many potential hiding spots: dense foliage, architectural details, busy patterns
- Rich colors and varied textures
- Elevated or bird's-eye view preferred

Scene types to vary between:
- Dense forest with intricate branches and leaves
- Busy marketplace with many stalls and objects  
- Complex architectural scene with ornate details
- Overgrown garden with tangled plants
- Cluttered workshop or laboratory
- Detailed cityscape from above

Output only the prompt, nothing else."""

        user_prompt = f"Generate a detailed image prompt for a {scene_type} scene perfect for hiding small objects."
        
        # Add to memory and get response
        messages = [
            HumanMessage(content=system_prompt + "\n\n" + user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        prompt = response.content.strip()
        
        # Store in memory
        self.memory.extend([
            HumanMessage(content=user_prompt),
            AIMessage(content=prompt)
        ])
        
        print(f"🎨 Agent generated prompt: {prompt[:100]}...")
        return prompt
    
    def _generate_image_with_hf(self, prompt: str) -> str:
        """
        Generate image using HuggingFace API with the provided prompt.
        """
        try:
            hf_api_key = os.getenv("HF_API_KEY")
            if not hf_api_key:
                raise Exception("HF_API_KEY not found in environment")
            
            # Use FLUX model for high quality
            url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
            
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            
            # Enhanced prompt with negative prompts
            enhanced_prompt = f"{prompt}, highly detailed, intricate patterns, complex scene, rich textures"
            
            data = {
                "inputs": enhanced_prompt,
                "parameters": {
                    "width": 1024,
                    "height": 1024,
                    "guidance_scale": 7.5,
                    "negative_prompt": "animals, people, human, capybara, creatures, faces, text, logo, watermark, low resolution, blurry"
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code != 200:
                raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
            
            # Save image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"capyfind/data/scene_{timestamp}.png"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"🎨 Agent saved image to: {output_path}")
            return output_path
            
        except Exception as e:
            error_msg = f"Image generation failed: {str(e)}"
            print(f"❌ Agent error: {error_msg}")
            raise Exception(error_msg)
    
    def generate_scene(self, scene_type: str = "random") -> ImageGenerationResult:
        """
        Main agent method: Generate a complete scene using LLM + image generation.
        
        This is the agent's primary capability that combines its tools.
        """
        print(f"🎨 ImageGeneratorAgent starting scene generation...")
        
        try:
            # Step 1: Agent creates a creative prompt using LLM
            prompt = self._create_image_prompt(scene_type)
            
            # Step 2: Agent generates the image
            image_path = self._generate_image_with_hf(prompt)
            
            # Step 3: Return structured result
            result = ImageGenerationResult(
                success=True,
                image_path=image_path,
                prompt_used=prompt,
                generation_time=datetime.now().isoformat()
            )
            
            print(f"✅ ImageGeneratorAgent completed successfully!")
            return result
            
        except Exception as e:
            error_result = ImageGenerationResult(
                success=False,
                error_message=str(e),
                generation_time=datetime.now().isoformat()
            )
            
            print(f"❌ ImageGeneratorAgent failed: {e}")
            return error_result
    
    def get_memory_summary(self) -> str:
        """Get a summary of the agent's memory/conversation history"""
        if not self.memory:
            return "No previous interactions"
        
        return f"Agent has {len(self.memory)} messages in memory. Last prompt was about: {self.memory[-1].content[:50]}..."


# Convenience function for backwards compatibility
def create_scene(scene_type: str = "random"):
    """
    Legacy function that creates an agent and generates a scene.
    Kept for compatibility with existing code.
    """
    agent = ImageGeneratorAgent()
    result = agent.generate_scene(scene_type)
    
    if result.success:
        return result.image_path, result.prompt_used
    else:
        raise Exception(result.error_message)
