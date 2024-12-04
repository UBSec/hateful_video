import sys
import json
import subprocess
import sys
import os
import csv
from vosk import Model, KaldiRecognizer, SetLogLevel
import tqdm.auto as tqdm

# This script is used to extract transcripts from audio


def extract_audio(video_path, audio_path):
"""
Converting mp4 to wav
@param video_path: path to the video file
@param audio_path: path of temporary audio file
@return: boolean value which determines if the extractions if successful
"""

    command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-ar', '16000', # SAMPLE RATE, Vosk use 16000
        '-ac', '1', # Number of audioi channel, 1 for Vosk model
        '-f', 'wav',
        audio_path
    ]
    try:
        subprocess.run(command,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    
def extract_videos_transcript(audio_path, rec):
"""
Extract transcript from audio and write into a csv file
@param audio_path: path to the temporary audio file
@param rec: recognizer
@return: transcript of a single audio 
"""
    transcript = ''
    with open(audio_path, 'rb') as audio_file:
        while True:
            data = audio_file.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get('text', '')
                transcript += text + ''
        final_piece = json.loads(rec.FinalResult())
        text = final_piece.get('text', '')
        transcript += text
        return transcript.strip()

def main():
    model_path = 'path/to/your/vosk_model'
    hate_videos_path = 'path/to/your/videos_or_audio'

    # video_names = ['hate_video_1', 'hate_video_2', 'hate_video_4', 'hate_video_6', 'hate_video_10', 'hate_video_13', 'hate_video_14', 'hate_video_15', 'hate_video_16', 'hate_video_17', 'hate_video_18', 'hate_video_21', 'hate_video_22', 'hate_video_23', 'hate_video_24', 'hate_video_25', 'hate_video_29', 'hate_video_33', 'hate_video_34', 'hate_video_36', 'hate_video_37', 'hate_video_41', 'hate_video_43', 'hate_video_45', 'hate_video_47', 'hate_video_48', 'hate_video_50', 'hate_video_52', 'hate_video_53', 'hate_video_55', 'hate_video_57', 'hate_video_58', 'hate_video_59', 'hate_video_62', 'hate_video_64', 'hate_video_68', 'hate_video_71', 'hate_video_72', 'hate_video_74', 'hate_video_79', 'hate_video_80', 'hate_video_81', 'hate_video_82', 'hate_video_83', 'hate_video_84', 'hate_video_88', 'hate_video_92', 'hate_video_99', 'hate_video_102', 'hate_video_108', 'hate_video_109', 'hate_video_111', 'hate_video_112', 'hate_video_113', 'hate_video_115', 'hate_video_117', 'hate_video_118', 'hate_video_120', 'hate_video_122', 'hate_video_123', 'hate_video_125', 'hate_video_126', 'hate_video_127', 'hate_video_129', 'hate_video_135', 'hate_video_136', 'hate_video_137', 'hate_video_138', 'hate_video_139', 'hate_video_141', 'hate_video_143', 'hate_video_146', 'hate_video_149', 'hate_video_151', 'hate_video_153', 'hate_video_156', 'hate_video_158', 'hate_video_162', 'hate_video_166', 'hate_video_167', 'hate_video_169', 'hate_video_173', 'hate_video_175', 'hate_video_177', 'hate_video_182', 'hate_video_184', 'hate_video_186', 'hate_video_187', 'hate_video_188', 'hate_video_190', 'hate_video_192', 'hate_video_194', 'hate_video_195', 'hate_video_196', 'hate_video_197', 'hate_video_198', 'hate_video_203', 'hate_video_205', 'hate_video_206', 'hate_video_209', 'hate_video_215', 'hate_video_217', 'hate_video_220', 'hate_video_223', 'hate_video_224', 'hate_video_225', 'hate_video_230', 'hate_video_233', 'hate_video_235', 'hate_video_236', 'hate_video_237', 'hate_video_240', 'hate_video_241', 'hate_video_243', 'hate_video_251', 'hate_video_253', 'hate_video_256', 'hate_video_257', 'hate_video_259', 'hate_video_260', 'hate_video_261', 'hate_video_271', 'hate_video_274', 'hate_video_275', 'hate_video_276', 'hate_video_280', 'hate_video_282', 'hate_video_283', 'hate_video_284', 'hate_video_286', 'hate_video_288', 'hate_video_291', 'hate_video_292', 'hate_video_295', 'hate_video_296', 'hate_video_299', 'hate_video_300', 'hate_video_302', 'hate_video_303', 'hate_video_308', 'hate_video_309', 'hate_video_310', 'hate_video_312', 'hate_video_313', 'hate_video_317', 'hate_video_318', 'hate_video_321', 'hate_video_322', 'hate_video_325', 'hate_video_326', 'hate_video_328', 'hate_video_330', 'hate_video_332', 'hate_video_333', 'hate_video_334', 'hate_video_339', 'hate_video_344', 'hate_video_345', 'hate_video_347', 'hate_video_349', 'hate_video_351', 'hate_video_352', 'hate_video_354', 'hate_video_358', 'hate_video_360', 'hate_video_361', 'hate_video_366', 'hate_video_367', 'hate_video_369', 'hate_video_371', 'hate_video_374', 'hate_video_376', 'hate_video_378', 'hate_video_379', 'hate_video_380', 'hate_video_383', 'hate_video_389', 'hate_video_390', 'hate_video_391', 'hate_video_393', 'hate_video_394', 'hate_video_396', 'hate_video_399', 'hate_video_401', 'hate_video_405', 'hate_video_406', 'hate_video_410', 'hate_video_412', 'hate_video_414', 'hate_video_415', 'hate_video_416', 'hate_video_418', 'hate_video_421', 'hate_video_422', 'hate_video_423', 'hate_video_425', 'hate_video_426', 'hate_video_427', 'hate_video_429', 'hate_video_430']
    output_csv = 'path/to/your/output.csv'
    audio_temp_dir = 'path/to/your/temporary_audio_save_folder'

    ## Configuration
    SAMPLE_RATE = 16000
    SetLogLevel(0)
    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    os.makedirs(audio_temp_dir, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as output_csv:
        csv_writer = csv.writer((output_csv))
        csv_writer.writerow(['Video Name', 'Transcript'])
        for video_name in tqdm.tqdm(video_names):
            video_path = os.path.join(hate_videos_path, f'{video_name}.mp4')
            audio_path = os.path.join(audio_temp_dir, f'{video_name}.wav')
            
            audio_is_available = extract_audio(video_path=video_path, audio_path=audio_path)
            if audio_is_available == False:
                csv_writer.writerow([f'{video_name}', 'No audio found'])
                continue
            elif os.path.getsize(audio_path) == 0:
                csv_writer.writerow([f'{video_name}', 'No audio found'])
                os.remove(audio_path)
                continue
            transcript = extract_videos_transcript(audio_path=audio_path, rec=rec)
            if not transcript:
                transcript = 'No audio found'
            csv_writer.writerow([f'{video_name}', f'{transcript}'])
            os.remove(audio_path)
if __name__ == '__main__':
    main()


