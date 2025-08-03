from __future__ import annotations
from livekit.agents import (
    Agent, 
    AgentSession, 
    JobContext, 
    RoomInputOptions,
    WorkerOptions,
    cli,
    )
from dotenv import load_dotenv
load_dotenv()
from prompts import AGENT_INSTRUCTION,SESSION_INSTRUCTION
from livekit.plugins import (
                            deepgram,
                            openai, 
                            cartesia, 
                            silero, 
                            noise_cancellation, 
                            # turn_detector
                            )
from livekit.plugins.turn_detector.multilingual import MultilingualModel
# from livekit.agents.stt import deepgram

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    # await ctx.wait_for_participant() auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL

    agent = Agent(instructions=AGENT_INSTRUCTION)
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM().with_ollama(model="llama3.2",temperature=0.8),
        tts=silero.TTS,
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(room=ctx.room, agent=agent,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )
    await session.generate_reply(instructions=SESSION_INSTRUCTION)

if __name__=="__main__":
    opts=WorkerOptions(entrypoint_fnc=entrypoint)
    cli.run_app(opts=opts)#LLM: ( llama3.2 || moondream )