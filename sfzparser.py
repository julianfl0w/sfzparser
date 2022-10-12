#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A parser for SFZ files."""

import math
import re
from pathlib import Path
import os
import sys
from collections import OrderedDict
import numpy as np
from io import open
import pickle
from tqdm import tqdm
import json
import librosa

SFZ_NOTE_LETTER_OFFSET = {"a": 9, "b": 11, "c": 0, "d": 2, "e": 4, "f": 5, "g": 7}


def sfz_note_to_midi_key(sfz_note, german=False):
    accidental = 0

    if "#" in sfz_note[1:] or "♯" in sfz_note:
        accidental = 1
    elif "b" in sfz_note[1:] or "♭" in sfz_note:
        accidental = -1

    letter = sfz_note[0].lower()
    if letter.isdigit():
        return sfz_note

    if german:
        # TODO: Handle sharps (e.g. "Fis") and flats (e.g. "Es")
        if letter == "b":
            accidental = -1
        if letter == "h":
            letter = "b"

    octave = int(sfz_note[-1])
    return max(
        0, min(127, SFZ_NOTE_LETTER_OFFSET[letter] + ((octave + 1) * 12) + accidental)
    )


def freq_to_cutoff(param):
    return 127.0 * max(0, min(1, math.log(param / 130.0) / 5)) if param else None


class SFZParser(object):
    rx_section = re.compile("^<([^>]+)>\s?")

    def __init__(self, sfz_path, encoding=None, **kwargs):
        self.encoding = encoding
        self.sfz_path = sfz_path
        self.groups = []
        self.sections = []

        with open(sfz_path, encoding=self.encoding or "utf-8-sig") as sfz:
            self.parse(sfz)

    def parse(self, sfz):
        section_name = ""
        sections = self.sections
        cur_section = []
        value = None

        for line in sfz:
            line = line.strip()

            if not line:
                continue

            if line.startswith("//"):
                sections.append(("comment", line))
                continue

            while line:
                match = self.rx_section.search(line)
                if match:
                    if cur_section:
                        sections.append(
                            (section_name, OrderedDict(reversed(cur_section)))
                        )
                        cur_section = []

                    section_name = match.group(1).strip()
                    line = line[match.end() :].lstrip()
                elif "=" in line:
                    line, _, value = line.rpartition("=")
                    if "=" in line:
                        line, key = line.rsplit(None, 1)
                        cur_section.append((key, value))
                        value = None
                elif value:
                    line, key = None, line
                    cur_section.append((key, value))
                else:
                    if line.startswith("//"):
                        print("Warning: inline comment")
                        sections.append(("comment", line))
                    # ignore garbage
                    break

        if cur_section:
            sections.append((section_name, OrderedDict(reversed(cur_section))))

        return sections


class SFZInstrument:
    def __init__(
        self,
        patch,
        platform_simple,
        SAMPLE_BUFFER_SIZE_SAMPLES,
        CHANNELS,
        SAMPLE_FREQUENCY,
    ):

        self.sfzFilename = patch.sfzFilename
        self.sfzFilenameBasedir = os.path.dirname(patch.sfzFilename)
        self.samplesBasedir = Path(patch.samplesBasedir).resolve()
        # see if we have already converted the wavs to necessary binary format

        # check if binary file has already been exported

        binFilename = os.path.join(
            self.sfzFilenameBasedir, "_" + platform_simple + ".bin"
        )
        if os.path.exists(binFilename):
            with open(binFilename, "rb") as f:
                self.sampleFilename2channelCount, self.sampleFilename2address, self.sampleFilename2lengthSamples, self.binaryBlob = pickle.load(
                    f
                )
        # otherwise create it!
        else:
            startAddr = 0
            # initialize the info
            self.sampleFilename2address = {}
            self.sampleFilename2lengthSamples = {}
            self.sampleFilename2channelCount = {}
            # binary blob contains only mono files
            # multitrack files will be broken apart and postpended
            # ex out.wav becomes out.wav0 and out.wav1
            self.binaryBlob = np.zeros((0), dtype=np.float32)

            self.allSampleFilenames = [
                str(s.resolve()) for s in list(Path(patch.samplesLoadPoint).rglob("*.wav"))
            ]
            self.allSampleFilenames += [
                str(s.resolve()) for s in list(Path(patch.samplesLoadPoint).rglob("*.flac"))
            ]
            # self.computeShader.sampleBuffer.zeroInitialize() # NO LONGER NECESSARY
            # load all the samples into a buffer
            for sampleFilename in tqdm(self.allSampleFilenames):
                # print("loading " + sampleFilename)

                y, samplerate = librosa.load(
                    sampleFilename,
                    sr=SAMPLE_FREQUENCY * 4,  # because shader access is 16byte
                )
                # addr = self.computeShader.sampleBuffer.write(y)
                # self.binaryBlob[startAddr : startAddr + len(y)] = y

                if len(np.shape(y)) == 1:
                    channelCount = 1
                else:
                    channelCount = np.shape(y)[1]

                self.sampleFilename2channelCount[sampleFilename] = channelCount
                for channel in range(channelCount):
                    self.binaryBlob = np.append(self.binaryBlob, y, axis=0)
                    sampleFilenameAndChannel = sampleFilename + "_" + str(channel)
                    self.sampleFilename2address[sampleFilenameAndChannel] = startAddr
                    self.sampleFilename2lengthSamples[sampleFilenameAndChannel] = len(y)
                startAddr += len(y)

            # save binary file for reloading
            with open(binFilename, "wb+") as f:
                pickle.dump(
                    (
                        self.sampleFilename2channelCount,
                        self.sampleFilename2address,
                        self.sampleFilename2lengthSamples,
                        self.binaryBlob,
                    ),
                    f,
                )

        ## save entire self for reloading
        # with open(binFilename, "wb+") as f:
        #    pkl.dump(self.sfzInst, f)

        self.loadSFZ(patch=patch)

    def preprocess(self, sfzFilename, replaceDict={}):
        print("processing file " + sfzFilename)
        # read in the template sfz
        with open(sfzFilename, "r") as f:
            preprocFile = f.read()

        preProcessText = ""
        for ogline in preprocFile.split("\n"):
            ogline = ogline.strip()
            line = ogline.split("//")[0]
            for k, v in replaceDict.items():
                line = line.replace(k, v)
                
            # keep the comments
            if ogline.startswith("//"):
                preProcessText += ogline

            if "#include" in line:
                includeParts = line.split("#include")[1:]
                print(includeParts)
                for i, includePart in enumerate(includeParts):
                    includePart = includePart.strip()
                    includeFilename = eval(includePart.split(" ")[0]) # NO SPACES ALLOWED IN FILENAMES!
                    includeFilename = os.path.join(
                        self.sfzFilenameBasedir, includeFilename
                    )
                    includeText = self.preprocess(
                        includeFilename, replaceDict=replaceDict.copy()
                    )
                    preProcessText += "\n" + includeText + "\n"

            elif line.startswith("#define"):
                preProcessText += line
                k = line.split(" ")[1]
                v = "".join(line.split(" ")[2:])
                replaceDict[k] = v
            else:
                preProcessText += line

            preProcessText += "\n"

        return preProcessText

    def loadSFZ(self, patch, depth=0):
        sfzFilename = patch.sfzFilename
        print("loading from " + sfzFilename)
        preProcessText = self.preprocess(sfzFilename)
        with open("a.spz", "w+") as f:
            f.write(preProcessText)

        sfzParser = SFZParser("a.spz")
        # pprint.pprint(sfzParser.sections)

        if depth == 0:
            self.regions = []
            self.globalDict = {}
            self.masterDict = {}
            self.groupDict = {}

        for sectionName, valueDict in sfzParser.sections:
            # print(sectionName)
            # print(valueDict)

            if sectionName == "region":
                # add all the global items
                for k, v in self.groupDict.items():  # GROUP
                    if k not in valueDict.keys():
                        valueDict[k] = v
                for k, v in self.masterDict.items():  # MASTER
                    if k not in valueDict.keys():
                        valueDict[k] = v
                for k, v in self.globalDict.items():  # GLOBAL
                    if k not in valueDict.keys():
                        valueDict[k] = v

                # resolve sample path
                resolved = str(
                    Path(
                        os.path.join(self.samplesBasedir, valueDict["sample"])
                    ).resolve()
                )
                valueDict["sample"] = resolved

                self.regions += [valueDict]

            elif sectionName == "global":
                self.globalDict = valueDict

            elif sectionName == "master":
                self.masterDict = valueDict

            elif sectionName == "group":
                self.groupDict = valueDict
                # print(json.dumps(valueDict, indent=2))

            elif sectionName == "control":
                pass

            elif sectionName == "comment":
                pass

            elif sectionName == "curve":
                pass
            elif sectionName == "":
                pass
            else:
                raise Exception("Unknown sfz header '" + str(sectionName) + "'")

        # attach salient regions to their midi note
        self.note2regions = []
        for i in range(128):
            thisNoteRegions = []
            for region in self.regions:
                if SFZInstrument.noteInSfzRegion(i, region):
                    thisNoteRegions += [region]
            self.note2regions += [thisNoteRegions]

    def noteInSfzRegion(noteNo, region):
        inRegion = True
        if "lokey" in region.keys() and noteNo < sfz_note_to_midi_key(region["lokey"]):
            inRegion = False
        if "hikey" in region.keys() and noteNo > sfz_note_to_midi_key(region["hikey"]):
            inRegion = False
        return inRegion


if __name__ == "__main__":
    import pprint
    import sys

    parser = SFZParser(sys.argv[1])
    pprint.pprint(parser.sections)
