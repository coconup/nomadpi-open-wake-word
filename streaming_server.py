# Imports
import aiohttp
from aiohttp import web
import numpy as np
from openwakeword import Model
import resampy
import argparse
import json

# Define websocket handler
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Send loaded models
    await ws.send_str(json.dumps({"loaded_models": list(owwModel.models.keys())}))

    # Start listening for websocket messages
    async for msg in ws:
        # Get the sample rate of the microphone from the browser
        if msg.type == aiohttp.WSMsgType.TEXT:
            sample_rate = int(msg.data)
        elif msg.type == aiohttp.WSMsgType.ERROR:
            print(f"WebSocket error: {ws.exception()}")
        else:
            # Get audio data from websocket
            audio_bytes = msg.data

            # Add extra bytes of silence if needed
            if len(msg.data) % 2 == 1:
                audio_bytes += (b'\x00')

            # Convert audio to correct format and sample rate
            data = np.frombuffer(audio_bytes, dtype=np.int16)

            try:
                if sample_rate != 16000:
                    data = resampy.resample(data, sample_rate, 16000)
            except UnboundLocalError:
                sample_rate = 16000

            # Get openWakeWord predictions and set to browser client
            predictions = owwModel.predict(data)

            activations = []
            for key in predictions:
                # print(f"Prediction {key}: {predictions[key]}")
                if predictions[key] >= 0.3:
                    activations.append(key)

            if activations != []:
                await ws.send_str(json.dumps({"activations": activations}))

    return ws

# Define static file handler
async def static_file_handler(request):
    return web.FileResponse('./streaming_client.html')

app = web.Application()
app.add_routes([web.get('/ws', websocket_handler), web.get('/', static_file_handler)])

if __name__ == '__main__':
    # Parse CLI arguments
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--chunk_size",
        help="How much audio (in number of samples) to predict on at once",
        type=int,
        default=1280,
        required=False
    )
    parser.add_argument(
        "--model_path",
        help="The path of a specific model to load",
        type=str,
        default="",
        required=False
    )
    parser.add_argument(
        "--inference_framework",
        help="The inference framework to use (either 'onnx' or 'tflite'",
        type=str,
        default='tflite',
        required=False
    )
    args=parser.parse_args()

    # Load openWakeWord models
    if args.model_path != "":
        owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework)
    else:
        owwModel = Model(inference_framework=args.inference_framework)

    # Start webapp
    web.run_app(app, host='0.0.0.0', port=9002)
