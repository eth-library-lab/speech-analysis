from os import fsdecode
from mlpm.solver import Solver
from deepspeech import Model
import wave
import shlex
import subprocess
import numpy as np
from textblob import TextBlob
import textblob
try:
    from shhlex import quote
except ImportError:
    from pipes import quote

def convert_sample_rate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

class speechSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        # Do you Init Work here
        self.ds = Model("pretrained/deepspeech-0.9.3-models.pbmm")
        self.scorepath = ("pretrained/deepspeech-0.9.3-models.scorer")
        self.ds.enableExternalScorer(self.scorepath)
        self.desired_sample_rate = self.ds.sampleRate()
        self.ready()
    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        # First convert audio file to wav format
        fin = wave.open(data['input_file_path'], 'rb')
        fs_orig = fin.getframerate()
        resampled = False
        if fs_orig != self.desired_sample_rate:
            resampled = True
            fs_new, audio = convert_sample_rate(data['input_file_path'], self.desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        
        audio_length = fin.getnframes() * (1/fs_orig)
        fin.close()
        
        result = self.ds.stt(audio)
        textblob_analyzer = TextBlob(result)
        sentiment = []
        for sentence in textblob_analyzer.sentences:
            sentiment.append({
                'sentence': str(sentence),
                'polarity': sentence.sentiment.polarity,
                'subjectivity': sentence.sentiment.subjectivity
            })
        return {
            "transcript": result,
            "audio_length": audio_length,
            "resampled": resampled,
            "sentiment": sentiment
        }