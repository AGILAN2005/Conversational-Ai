from __future__ import annotations
from dotenv import load_dotenv
import os, logging
from livekit import agents
from livekit.agents import JobContext, WorkerOptions, cli, llm
from livekit.agents.multimodal import MultimodalAgent, AgentTranscriptionOptions
from livekit.plugins import openai

from api import AssistantFnc
from prompts import WELCOME_MESSAGE, INSTRUCTIONS, LOOKUP_VIN_MESSAGE

load_dotenv()

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

async def entrypoint(ctx: JobContext):
    logger.info("Connecting to room")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    model = openai.realtime.RealtimeModel(
        instructions=INSTRUCTIONS,
        voice="shimmer",
        temperature=0.8,
        modalities=["audio", "text"],
        max_response_output_tokens=2048,
    )

    assistant = MultimodalAgent(
        model=model,
        transcription=AgentTranscriptionOptions(
            user_transcription=True,
            agent_transcription=True,
            agent_transcription_speed=1.0
        )
    )

    assistant.start(ctx.room)
    session = model.sessions[0]

    # Send welcome prompt
    session.conversation.item.create(
        llm.ChatMessage(role="assistant", content=WELCOME_MESSAGE)
    )
    session.response.create()

    assistant_fnc = AssistantFnc()

    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        content = msg.content if isinstance(msg.content, str) else "\n".join(
            "[image]" if isinstance(x, llm.ChatImage) else x
            for x in msg.content
        )

        if assistant_fnc.has_car():
            session.conversation.item.create(llm.ChatMessage(role="user", content=content))
        else:
            session.conversation.item.create(
                llm.ChatMessage(role="system", content=LOOKUP_VIN_MESSAGE(content))
            )

        session.response.create()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
