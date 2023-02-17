from pytube import YouTube
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import itertools

LINK_YOUTUBE = ""
OUPUT_AUDIOS = ""
OUPTUT_DIR = ""
SUFFIX_NAME = ""


def split_on_silence_improved(audio_segment, min_silence_len=300, silence_thresh=-50, keep_silence=200,
                              seek_step=1):
    """
    audio_segment - original pydub.AudioSegment() object
    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms
    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS
    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms
    """

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    not_silence_ranges = detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)

    # from the itertools documentation
    def pairwise(iterable):
        """
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        :param iterable:
        :return:
        """
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    start_min = 0
    start_ii = 0
    end_ii = 0
    chunks = []
    moments = []
    for (start_i, end_i), (start_ii, end_ii) in pairwise(not_silence_ranges):
        end_max = end_i + (start_ii - end_i + 1) // 2  # +1 for rounding with integer division
        start_i = max(start_min, start_i - keep_silence)
        end_i = min(end_max, end_i + keep_silence)

        chunks.append(audio_segment[start_i:end_i])
        start_min = end_max
        moments.append((start_i, end_i))
    chunks.append(audio_segment[max(start_min, start_ii - keep_silence):
                                min(len(audio_segment), end_ii + keep_silence)])

    moments.append((max(start_min, start_ii - keep_silence), min(len(audio_segment), end_ii + keep_silence)))
    return chunks, moments


def main():
    yt = YouTube(LINK_YOUTUBE)
    stream = yt.streams.filter(mime_type="audio/webm", only_audio=True)[0]
    path_download = stream.download("output_audios/", filename="pique_enfadado.webm")
    audio = AudioSegment.from_file(path_download)
    audio, _ = audio.split_to_mono()

    chunks, moments = split_on_silence_improved(audio, silence_thresh=-50, min_silence_len=300)

    for i, cc in enumerate(chunks):
        cc = cc.set_frame_rate(16000)
        output_path = os.path.join(OUPTUT_DIR, SUFFIX_NAME + "_" + str(i) + ".mp3")
        cc.export(out_f=output_path, format="mp3")


if __name__ == "__main__":
    main()
