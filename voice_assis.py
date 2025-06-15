from __future__ import annotations
from livekit.agents import (
    Agent, 
    AgentSession, 
    JobContext, 
    RoomInputOptions,
    WorkerOptions,
    cli,
    AutoSubscribe
    )
from dotenv import load_dotenv
load_dotenv()
from prompts import INSTRUCTIONS,WELCOME_MESSAGE
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

    agent = Agent(instructions=INSTRUCTIONS)
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM().with_ollama(model="llama3.2",temperature=0.8),
        tts=cartesia.TTS(voice="shimmer"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(room=ctx.room, agent=agent,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )
    await session.generate_reply(instructions=WELCOME_MESSAGE)

if __name__=="__main__":
    opts=WorkerOptions(entrypoint_fnc=entrypoint)
    cli.run_app(opts=opts)#LLM: ( llama3.2 || moondream )