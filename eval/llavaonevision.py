import torch
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import av
import cv2
import copy

warnings.filterwarnings("ignore")

# Paths to the folders containing hate and non-hate videos
hate_video_folder = '/home/shared/Data/HateMM/hate_videos'
non_hate_folder = '/home/shared/Data/HateMM/non_hate_videos'
labels_map = {'hate': 1, 'non_hate': 0}

def read_video_frames(video_path, frames_target=24, frame_size=(224, 224)):
    """
    Reads and processes frames from a video file.

    Parameters:
    - video_path (str): Path to the video file.
    - frames_target (int): Desired number of frames to extract.
    - frame_size (tuple): Desired frame size as (width, height).

    Returns:
    - sampled_frames (list): List of processed frames as numpy arrays.
    """
    try:
        # Open the video file using PyAV
        container = av.open(video_path)
        videostream = container.streams.video[0]
        average_fps = float(videostream.average_rate)
        duration_in_seconds = float(container.duration / av.time_base)
        total_seconds = int(duration_in_seconds)

        # Calculate sample indices based on the video's FPS and duration
        sample_indices = [int(second * average_fps) for second in range(total_seconds)]
        sample_indices_set = set(sample_indices)
        sampled_frames = []

        # Decode the video and collect frames at the sample indices
        for index, frame in enumerate(container.decode(video=0)):
            if index in sample_indices_set:
                image_frame = frame.to_ndarray(format='rgb24')
                image_frame = cv2.resize(image_frame, frame_size)
                sampled_frames.append(image_frame)
            if index > sample_indices[-1]:
                break
        container.close()

        number_of_frames = len(sampled_frames)

        # Adjust the number of frames to match the target
        if number_of_frames < frames_target:
            # Pad with white frames if not enough frames
            frame_shape = sampled_frames[0].shape if number_of_frames > 0 else (*frame_size, 3)
            padding_sequence = [np.ones(frame_shape, dtype=np.uint8) * 255] * (frames_target - number_of_frames)
            sampled_frames.extend(padding_sequence)
        elif number_of_frames > frames_target:
            # Uniformly sample frames to reduce to the target number
            uniform_sampling_indices = np.linspace(0, number_of_frames - 1, frames_target).astype(int)
            sampled_frames = [sampled_frames[index] for index in uniform_sampling_indices]

        return sampled_frames[:frames_target]

    except Exception as e:
        print(f"Error handling {video_path}: {e}")
        # Return white frames in case of an error
        frame_shape = (*frame_size, 3)
        sampled_frames = [np.ones(frame_shape, dtype=np.uint8) * 255] * frames_target
        return sampled_frames

def process_videos(gpu_id, all_video_filepaths, true_labels, results_queue):
    """
    Processes a list of videos using the Llava OneVision model on a specified GPU.

    Parameters:
    - gpu_id (int): GPU identifier to use for processing.
    - all_video_filepaths (list): List of video file paths to process.
    - true_labels (list): Corresponding true labels for the videos.
    - results_queue (multiprocessing.Queue): Queue to store processing results.
    """
    import os
    import numpy as np
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    import string

    # Set the GPU device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model and tokenizer
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    llava_model_args = {
        'multimodal': True,
    }
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=None, attn_implementation="sdpa", **llava_model_args
    )

    model = model.to(device)
    model.eval()

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    local_results = []

    # Process each video
    for index, video_path in enumerate(all_video_filepaths):
        print(f'GPU {gpu_id}: Processing video {os.path.basename(video_path).strip()}, progress: {(index + 1) / len(all_video_filepaths):.2f}')
        try:
            # Read and preprocess video frames
            video_frames = read_video_frames(video_path, frames_target=24, frame_size=(224, 224))
            video_frames = np.array(video_frames)
            frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(device)

            # Construct the question prompt
            question = f"{DEFAULT_IMAGE_TOKEN}\nIs there any hateful content in this video? Answer 'Yes' or 'No' and must explain why."
            conv_template = "qwen_1_5"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            # Tokenize input and prepare attention mask
            encoded_inputs = tokenizer_image_token(
                prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            input_ids = encoded_inputs.unsqueeze(0).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()

            # Generate model output
            cont = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=[frames],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                modalities=["video"],
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            model_response = text_outputs[0].split('assistant')[-1].strip()

            # Extract the predicted label from the response
            response = model_response.strip().lower()
            response = response.strip(string.whitespace + string.punctuation + '\"\'')
            first_word = response.split()[0] if response else 'unknown'
            first_word = first_word.strip(string.punctuation)

            if 'yes' in first_word:
                predicted_label = 'hate'
            elif 'no' in first_word:
                predicted_label = 'non_hate'
            else:
                predicted_label = 'unknown'

            # Append the result to the local results
            local_results.append(
                {
                    'video_name': os.path.basename(video_path).strip(),
                    'true_label': true_labels[index],
                    'predicted_label': predicted_label,
                    'model_response': model_response,
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1_score': np.nan
                }
            )

        except Exception as e:
            print(f'GPU {gpu_id}: Error processing video {os.path.basename(video_path).strip()} as {e}')
            # Handle errors gracefully
            local_results.append(
                {
                    'video_name': os.path.basename(video_path).strip(),
                    'true_label': true_labels[index],
                    'predicted_label': 'error',
                    'model_response': f'Error:{e}',
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1_score': np.nan
                }
            )
        finally:
            torch.cuda.empty_cache()
    # Put the local results into the shared queue
    results_queue.put(local_results)

if __name__ == '__main__':
    # Set the start method for multiprocessing
    mp.set_start_method('spawn', force=True)

    # Collect video file paths and labels
    hate_video_filepaths = [os.path.join(hate_video_folder, filename) for filename in os.listdir(hate_video_folder)]
    non_hate_video_filepaths = [os.path.join(non_hate_folder, filename) for filename in os.listdir(non_hate_folder)]
    all_video_filepaths = hate_video_filepaths + non_hate_video_filepaths
    true_labels = ['hate'] * len(hate_video_filepaths) + ['non_hate'] * len(non_hate_video_filepaths)

    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Split videos and labels among GPUs
    video_splits = [[] for _ in range(num_gpus)]
    label_splits = [[] for _ in range(num_gpus)]
    for index, (video, truelabel) in enumerate(zip(all_video_filepaths, true_labels)):
        gpu_id = index % num_gpus
        video_splits[gpu_id].append(video)
        label_splits[gpu_id].append(truelabel)

    for gpu_id in range(num_gpus):
        print(f"GPU {gpu_id} has {len(video_splits[gpu_id])} videos assigned.")

    # Create a multiprocessing queue to collect results
    results_queue = mp.Queue()
    processes_collection = []

    # Start processing videos on each GPU
    for gpu_id in range(num_gpus):
        process = mp.Process(target=process_videos, args=(gpu_id, video_splits[gpu_id], label_splits[gpu_id], results_queue))
        processes_collection.append(process)
        process.start()

    testing_results = []
    # Collect results from all processes
    for _ in range(num_gpus):
        testing_results.extend(results_queue.get())
    for process in processes_collection:
        process.join()

    # Create a DataFrame to store results
    resultsdf = pd.DataFrame(testing_results)
    # Filter out invalid results
    valid_results = [result for result in testing_results if result['predicted_label'] in labels_map]
    y_true = [labels_map[result['true_label']] for result in valid_results]
    y_predict = [labels_map[result['predicted_label']] for result in valid_results]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    metrics = {
        "video_name": "Overall Metrics",
        "true_label": "",
        "predicted_label": "",
        "model_response": "",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    # Append overall metrics to the DataFrame
    resultsdf = pd.concat([resultsdf, pd.DataFrame([metrics])], ignore_index=True)
    # Save results to a CSV file
    resultsdf.to_csv('llava_onevision_onHateMM_results.csv', index=False)
