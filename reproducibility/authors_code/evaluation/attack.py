# Adapted from VRP:

import torch
from PIL import Image, ImageFont, ImageDraw
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers import AutoModel, CLIPImageProcessor
from transformers import pipeline
import transformers
import torchvision.transforms as T

import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import base64


def main(args):
    # Open image folder

    save_path = args.save_path
    model_name = args.model_name
    attack_root = args.attack_root
    image_root = args.image_root
    attack_config = args.attack_config

    os.makedirs(os.path.dirname(f"{save_path}/{attack_config}/{model_name}.csv"), exist_ok=True)    
    query_df = pd.read_csv(f"{attack_root}/{attack_config}.csv")

    print("Generating "+f"{save_path}/{model_name}/{attack_config}.csv")
    batch_query_text = query_df["text"]
    batch_image_path =  [ f"{image_root}/{image_file}" if image_file and not pd.isna(image_file) else None for image_file in query_df["image"]] 

    batch_response = [None] * len(batch_image_path) 
    

    if model_name == 'llava-hf/llava-v1.6-mistral-7b-hf':
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        processor = LlavaNextProcessor.from_pretrained(model_name)

        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16) #, low_cpu_mem_usage=True) 
        model.to("cuda")
        
    
        for index, (image_path, prompt) in enumerate(zip(batch_image_path, batch_query_text)):
            prompt = f"[INST] <image>\n{prompt}[/INST]"
            if image_path:
                print(image_path)
                image = Image.open(image_path)
            else:
                image = None
            inputs = processor(image, prompt, return_tensors="pt").to("cuda")
            output = model.generate(**inputs, max_new_tokens=512)
            response = processor.decode(output[0], skip_special_tokens=True)
            batch_response[index] = response
            query_df["response"] = batch_response
            print(response)
        query_df.to_csv(f"{save_path}/{attack_config}/{model_name}.csv")
    
    elif model_name == "Qwen/Qwen-VL-Chat":
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

        for index, (image_path, prompt) in enumerate(zip(batch_image_path, batch_query_text)):
            if image_path:
                response = model.chat(tokenizer, query=f'<img>{image_path}</img>{prompt}', history=None)
            else:
                response = model.chat(tokenizer, query=f'{prompt}', history=None)
            batch_response[index] = response[0]
            query_df["response"] = batch_response
            print(response[0])
        query_df.to_csv(f"{save_path}/{attack_config}/{model_name}.csv")

    elif model_name == "Qwen/Qwen2-VL-7B-Instruct":
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
                
        # default: Load the model on the available device(s)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
          "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
           
        # default processer
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        for index, (image_path, prompt) in enumerate(zip(batch_image_path, batch_query_text)):
            print(image_path)
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Inference: Generation of the output
            outputs = model.generate(**inputs, max_new_tokens=512)
            decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
            print(decoded_output)
            batch_response[index] = decoded_output
            query_df["response"] = batch_response
    
            

        query_df.to_csv(f"{save_path}/{attack_config}/{model_name}.csv")


    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info
                
        # default: Load the model on the available device(s)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
          "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
           
        # default processer
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print(f"Model config: {model.config}")
        print(f"Model dtype: {model.dtype}")
        print(f"Training mode: {model.training}") 
        for index, (image_path, prompt) in enumerate(zip(batch_image_path, batch_query_text)):
            print(image_path)
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
           
        
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Inference: Generation of the output
            outputs = model.generate(**inputs, max_new_tokens=512)
            decoded_output = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
            print(decoded_output)
            batch_response[index] = decoded_output
            query_df["response"] = batch_response
    
            

        query_df.to_csv(f"{save_path}/{attack_config}/{model_name}.csv")

    elif model_name == "THUDM/GLM-4.1V-9B-Thinking":
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking")
        model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking",
            torch_dtype="auto",
            device_map="auto",
        )

        for index, (image_path, prompt) in enumerate(zip(batch_image_path, batch_query_text)):
            # Build messages in GLM's expected format
            if image_path:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
           
        
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"data:image/png;base64,{base64_image}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            else:
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

            # Prepare inputs using chat template
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

            # Remove token_type_ids if present (in case of tokenizer mismatch)
            inputs.pop("token_type_ids", None)

            # Generate output
            outputs = model.generate(**inputs, max_new_tokens=512)

            # Decode only new tokens after input length
            response = processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            )

            batch_response[index] = response
            print(response)

        query_df["response"] = batch_response
        query_df.to_csv(f"{save_path}/{attack_config}/{model_name}.csv")


   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument("--attack_config", type=str, default="test")
    
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--attack_root", type=str, default="./attack_configs")
    parser.add_argument("--image_root", type=str, default=".")

    args = parser.parse_args()


    main(args)
    