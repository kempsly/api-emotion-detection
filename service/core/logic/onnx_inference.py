# # import onnx_runtime as rt
# import onnxruntime as rt
# import cv2 
# import numpy as np


# def emotions_detector(img_array):
    
#     if len(img_array.shape) == 2:
#         img_array =  cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
#     providers=['CPUExecutionProvider' ]
#     m_q = rt.InferenceSession(
#         "eff_quantized.onnx", 
#          providers=providers
#          )
    
#     print(m_q.get_inputs())
#     test_image = cv2.resize(img_array, (256, 256))
#     im = np.float32(test_image)
#     img_array = np.expand_dims(im, axis = 0)
    
#     onnx_pred = m_q.run(['dense'], {"input":img_array})
#     print(np.argmax(onnx_pred[0][0]))
    
#     emotion =""
#     if np.argmax(onnx_pred[0][0]) == 0:
#         emotion = "angry"
#     elif np.argmax(onnx_pred[0][0]) == 1:
#         emotion = "happy"
#     else:
#         emotion ="sad"
    
    
#     return {"emotion": emotion}

import onnxruntime as rt
import cv2 
import numpy as np
import time

# import service.main as s


def emotions_detector(img_array):
    # Preprocess input image if needed (e.g., convert grayscale to RGB)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    time_init = time.time()
    
    # Load the ONNX model
    model_path = "eff_quantized.onnx"
    providers = ['CPUExecutionProvider']
    m_q = rt.InferenceSession(model_path, providers=providers)
    
    
    # Resize and prepare input data
    test_image = cv2.resize(img_array, (256, 256))
    img_array = np.float32(test_image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # time_elapsed_preprocess = time.time() - time_init
    
    # Get input node name
    input_name = m_q.get_inputs()[0].name  # Assuming single input
    
    # Run inference
    onnx_pred = m_q.run(None, {input_name: img_array})
    
    time_elapsed = time.time() - time_init
    
    # Process the output
    class_idx = np.argmax(onnx_pred)
    
    # Map class index to emotion label
    if class_idx == 0:
        emotion = "angry"
    elif class_idx == 1:
        emotion = "happy"
    else:
        emotion = "sad"
    
    return {
        "emotion": emotion,
        "time_elapsed": str(time_elapsed),
        # "time_elapsed_preprocess": str(time_elapsed_preprocess)
        }
