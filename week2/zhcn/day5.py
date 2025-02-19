import os
import json
import logging
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import threading
from io import BytesIO
import base64

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from pydub import AudioSegment
from pydub.playback import play

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
from dataclasses import dataclass, field

@dataclass
class Config:
    MODEL: str = 'gpt-4o-mini'
    VOICE_MODEL: str = 'tts-1'
    VOICE_TYPE: str = 'alloy'
    IMAGE_MODEL: str = 'dall-e-3'
    MAX_HISTORY: int = 50
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant for an Airline called FlightAI. "
        "Give short, courteous answers, no more than 1 sentence. "
        "Always be accurate. If you don't know the answer, say so."
    )
    CITY_TICKET_PRICES: Dict[str, str] = field(
        default_factory=lambda: {
            "london": "$799",
            "paris": "$999",
            "tokyo": "$1099",
        }
    )

config = Config()

# Environment setup
def validate_env() -> None:
    """Validate required environment variables."""
    required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

load_dotenv(override=True)
validate_env()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

class ChatHistory:
    """Manage chat history with size limit."""
    def __init__(self, max_length: int = config.MAX_HISTORY):
        self.messages: List[Dict[str, str]] = []
        self.max_length = max_length

    def add_message(self, role: str, content: str) -> None:
        """Add a message to history with size limit enforcement."""
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_length:
            self.messages.pop(0)

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in history."""
        return self.messages

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages = []

class FlightAIAssistant:
    """Main assistant class handling all AI interactions."""
    def __init__(self):
        self.chat_history = ChatHistory()
        self.setup_tools()

    def setup_tools(self) -> None:
        """Setup available tools for the assistant."""
        get_price_tool_spec = {
            "name": "get_flight_ticket_price",
            "description": "Provides the price of a return flight ticket to the specified destination city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "The city the user wants to fly to.",
                    },
                },
                "required": ["destination_city"],
            },
        }
        self.available_tools = [
            {"type": "function", "function": get_price_tool_spec},
        ]

    @staticmethod
    @lru_cache(maxsize=100)
    def get_flight_ticket_price(destination_city: str) -> str:
        """Get ticket price for a city with caching."""
        return config.CITY_TICKET_PRICES.get(destination_city.lower(), "Price Not Found")

    def generate_image(self, city: str) -> Optional[Image.Image]:
        """Generate an image for a city."""
        try:
            image_response = client.images.generate(
                model=config.IMAGE_MODEL,
                prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
                n=1,
                response_format="b64_json"
            )
            image_base64 = image_response.data[0].b64_json
            image_data = base64.b64decode(image_base64)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return None

    def text_to_speech(self, message: str) -> None:
        """Convert text to speech and play asynchronously."""
        try:
            response = client.audio.speech.create(
                model=config.VOICE_MODEL,
                voice=config.VOICE_TYPE,
                input=message
            )
            audio_stream = BytesIO(response.content)
            audio = AudioSegment.from_file(audio_stream, format="mp3")
            threading.Thread(target=play, args=(audio,), daemon=True).start()
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")

    def handle_tool_call(self, message: Any) -> Tuple[Dict[str, str], str]:
        """Handle tool calls from the AI."""
        tool_call = message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get('destination_city')
        price = self.get_flight_ticket_price(city)
        response = {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id
        }
        return response, city

    def chat(self, history: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Optional[Image.Image]]:
        """Process chat messages and generate responses."""
        try:
            messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + history
            response = client.chat.completions.create(
                model=config.MODEL,
                messages=messages,
                tools=self.available_tools
            )
            
            image = None
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_response, city = self.handle_tool_call(message)
                messages.append(message)
                messages.append(tool_response)
                image = self.generate_image(city)
                response = client.chat.completions.create(
                    model=config.MODEL,
                    messages=messages
                )

            reply = response.choices[0].message.content
            history += [{"role": "assistant", "content": reply}]
            self.text_to_speech(reply)
            return history, image

        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            error_message = {"role": "assistant", "content": "I apologize, but I encountered an error. Please try again."}
            history += [error_message]
            return history, None

def create_ui() -> gr.Blocks:
    """Create and configure the Gradio UI."""
    assistant = FlightAIAssistant()

    def do_entry(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
        """Handle new message entry."""
        history += [{"role": "user", "content": message}]
        return "", history

    with gr.Blocks(theme=gr.themes.Soft()) as ui:
        gr.Markdown("# FlightAI Assistant")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    height=500,
                    type="messages",
                    show_label=False
                )
            with gr.Column(scale=1):
                image_output = gr.Image(
                    height=500,
                    label="Destination Preview"
                )
        
        with gr.Row():
            entry = gr.Textbox(
                label="Chat with our AI Assistant:",
                placeholder="Type your message here..."
            )
        
        with gr.Row():
            clear = gr.Button("Clear Chat")

        # Set up event handlers
        entry.submit(
            do_entry,
            inputs=[entry, chatbot],
            outputs=[entry, chatbot]
        ).then(
            assistant.chat,
            inputs=chatbot,
            outputs=[chatbot, image_output]
        )

        clear.click(
            lambda: (None, None),
            inputs=None,
            outputs=[chatbot, image_output],
            queue=False
        )

    return ui

if __name__ == "__main__":
    try:
        ui = create_ui()
        ui.launch(inbrowser=True)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")