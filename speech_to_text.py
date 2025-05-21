import os
import gc
import json
import re
import torch
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment
from datetime import datetime
from typing import List, Dict, Optional
from pyannote.audio import Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import torchaudio
import whisper
import difflib

# Load .env variables
load_dotenv()

CONFIG = {
    "WHISPER_MODEL": "large-v3",
    "DIARIZATION_MODEL": "pyannote/speaker-diarization-3.1",
    "TEXT_ENHANCEMENT_MODEL": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
    "DEEPL_API_KEY": os.getenv("DEEPL_API_KEY", ""),
    "SAMPLE_RATE": 16000,
    "NOISE_REDUCTION_LEVEL": 0.2,
    "MIN_SPEAKER_DURATION": 0.5,
    "MIN_AUDIO_LENGTH": 1.0,
    "MAX_SPEAKERS": 5,
    "ENHANCE_TEMPERATURE": 0.7,
    "ENHANCE_TOP_P": 0.9,
    "ENHANCE_MAX_NEW_TOKENS": 512,
    "TARGET_LANGUAGE": "EN-US",
    "USE_GPU": torch.cuda.is_available(),
    "MAX_GPU_MEMORY": 0.9,
    "OUTPUT_DIR": "outputs"
}

class AdvancedSpeechToText:
    def __init__(self):
        self.device = torch.device("cuda" if CONFIG["USE_GPU"] else "cpu")
        self.models = {}
        self.translator = None
        self._setup_directories()
        self._validate_tokens()

    def _validate_tokens(self):
        if not CONFIG["HF_TOKEN"]:
            raise ValueError("Hugging Face token is required")
        if not CONFIG["DEEPL_API_KEY"]:
            print("\u26a0\ufe0f DeepL API key not found - translation will be skipped")

    def _setup_directories(self):
        os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
        os.makedirs(f"{CONFIG['OUTPUT_DIR']}/speakers", exist_ok=True)
        os.makedirs(f"{CONFIG['OUTPUT_DIR']}/transcripts", exist_ok=True)

    def _validate_audio(self, audio_path: str) -> bool:
        info = sf.info(audio_path)
        if info.duration < CONFIG["MIN_AUDIO_LENGTH"]:
            raise ValueError(f"Audio too short: {info.duration:.2f}s")
        return True

    def _load_audio(self, audio_path: str) -> np.ndarray:
        audio, orig_sr = sf.read(audio_path, dtype='float32')
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if orig_sr != CONFIG["SAMPLE_RATE"]:
            resampler = torchaudio.transforms.Resample(orig_sr, CONFIG["SAMPLE_RATE"])
            audio_tensor = torch.from_numpy(audio).float()
            audio = resampler(audio_tensor).numpy()
        return audio

    def _preprocess_audio(self, audio_path: str) -> np.ndarray:
        self._validate_audio(audio_path)
        audio = self._load_audio(audio_path)
        return nr.reduce_noise(y=audio, sr=CONFIG["SAMPLE_RATE"], stationary=True, prop_decrease=CONFIG["NOISE_REDUCTION_LEVEL"])

    def _run_diarization(self, audio_path: str):
        if "diarization" not in self.models:
            self.models["diarization"] = Pipeline.from_pretrained(CONFIG["DIARIZATION_MODEL"], use_auth_token=CONFIG["HF_TOKEN"]).to(self.device)
        return self.models["diarization"](audio_path, min_speakers=1, max_speakers=CONFIG["MAX_SPEAKERS"])

    def _process_diarization(self, diarization) -> List[dict]:
        return [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
                "duration": float(turn.end - turn.start)
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
            if (turn.end - turn.start) >= CONFIG["MIN_SPEAKER_DURATION"]
        ]

    def _group_speakers(self, segments: List[dict]) -> Dict[str, List[dict]]:
        grouped = {}
        for seg in segments:
            grouped.setdefault(seg["speaker"], []).append(seg)
        return grouped

    def _create_speaker_audio(self, original_path: str, segments: List[dict], speaker: str) -> str:
        full_audio = AudioSegment.silent(duration=0)
        audio = AudioSegment.from_file(original_path)
        for seg in segments:
            full_audio += audio[int(seg["start"]*1000):int(seg["end"]*1000)]
        out_path = f"{CONFIG['OUTPUT_DIR']}/speakers/{speaker}.wav"
        full_audio.export(out_path, format="wav")
        return out_path

    def _clean_transcript(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)
        return text[0].upper() + text[1:] if text else text

    def _transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> dict:
        if "whisper" not in self.models:
            self.models["whisper"] = whisper.load_model(CONFIG["WHISPER_MODEL"], device=self.device)
        result = self.models["whisper"].transcribe(audio_path, language=language) if language else self.models["whisper"].transcribe(audio_path)
        return {
            "raw_text": result["text"],
            "cleaned_text": self._clean_transcript(result["text"])
        }

    def _load_text_enhancer(self):
        if "text_enhancer" not in self.models:
            tokenizer = AutoTokenizer.from_pretrained(CONFIG["TEXT_ENHANCEMENT_MODEL"])
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG["TEXT_ENHANCEMENT_MODEL"],
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_4bit=True
            )
            self.models["text_enhancer"] = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto"
            )

    def _strip_notes(self, text: str) -> str:
        return re.split(r'\bNote\b:|\bAdditional context\b:|\bExplanation\b:', text)[0].strip()

    def _enhance_text(self, text: str) -> str:
        self._load_text_enhancer()
        prompt = f"""Improve the following transcription and finish missing words or letters to be more close to the speaker intent. without adding explanations, notes, or comments. Output only the enhanced text.\n\nOriginal: \"{text}\"\nEnhanced:"""
        output = self.models["text_enhancer"](
            prompt,
            temperature=CONFIG["ENHANCE_TEMPERATURE"],
            top_p=CONFIG["ENHANCE_TOP_P"],
            max_new_tokens=CONFIG["ENHANCE_MAX_NEW_TOKENS"],
            do_sample=True,
            pad_token_id=self.models["text_enhancer"].tokenizer.eos_token_id
        )[0]['generated_text']
        match = re.search(r"Enhanced:(.*)", output, re.DOTALL)
        enhanced = self._clean_transcript(match.group(1).strip()) if match else text
        return self._strip_notes(enhanced)

    def _calculate_accuracy(self, reference: str, hypothesis: str) -> float:
        ref = reference.lower().split()
        hyp = hypothesis.lower().split()
        seq = difflib.SequenceMatcher(None, ref, hyp)
        return round(seq.ratio(), 4)

    def _translate_text(self, text: str) -> str:
        if not CONFIG["DEEPL_API_KEY"]:
            return "Translation not available"
        try:
            import deepl
            if not self.translator:
                self.translator = deepl.Translator(CONFIG["DEEPL_API_KEY"])
            result = self.translator.translate_text(text, target_lang=CONFIG["TARGET_LANGUAGE"]).text
            return self._strip_notes(self._clean_transcript(result))
        except Exception as e:
            return f"Translation error: {str(e)}"

    def process_audio(self, audio_path: str, language: Optional[str] = None) -> List[dict]:
        if CONFIG["USE_GPU"]:
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(CONFIG["MAX_GPU_MEMORY"])

        try:
            audio = self._preprocess_audio(audio_path)
            diarization = self._run_diarization(audio_path)
            segments = self._process_diarization(diarization)
            speakers = self._group_speakers(segments)
            results = []

            self.models["whisper"] = whisper.load_model(CONFIG["WHISPER_MODEL"], device=self.device)

            for speaker, segs in speakers.items():
                speaker_audio = self._create_speaker_audio(audio_path, segs, speaker)
                transcript = self._transcribe_audio(speaker_audio, language)
                enhanced = self._enhance_text(transcript["cleaned_text"])
                translation = self._translate_text(enhanced)
                accuracy = self._calculate_accuracy(transcript["raw_text"], enhanced)

                result = {
                    "speaker_id": speaker,
                    "audio_file": speaker_audio,
                    "total_duration": sum(s["duration"] for s in segs),
                    "segment_count": len(segs),
                    "segments": segs,
                    "transcripts": {
                        "raw": transcript["raw_text"],
                        "cleaned": transcript["cleaned_text"],
                        "enhanced": enhanced,
                        "transcription_accuracy": accuracy,
                        "translation": translation,
                        "target_language": CONFIG["TARGET_LANGUAGE"]
                    },
                    "processing_date": datetime.now().isoformat()
                }

                with open(f"{CONFIG['OUTPUT_DIR']}/transcripts/{speaker}.json", 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                results.append(result)

            with open(f"{CONFIG['OUTPUT_DIR']}/combined_results.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "audio_file": audio_path,
                    "speakers": [r["speaker_id"] for r in results],
                    "results": results,
                    "status": "complete",
                    "processed_date": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)

            return results

        except Exception as e:
            return [{"status": "failed", "error": str(e)}]

        finally:
            self.models.clear()
            if self.translator:
                del self.translator
            torch.cuda.empty_cache()
            gc.collect()
