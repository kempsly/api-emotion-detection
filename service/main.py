from typing import Union
from fastapi import FastAPI
import onnxruntime as rt

from service.api.api import main_router




app = FastAPI(projet_name="Emotions Detection")
app.include_router(main_router)

# # Load the ONNX model
# model_path = "eff_quantized.onnx"
# providers = ['CPUExecutionProvider']
# m_q = rt.InferenceSession(model_path, providers=providers)


@app.get("/")
async def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
