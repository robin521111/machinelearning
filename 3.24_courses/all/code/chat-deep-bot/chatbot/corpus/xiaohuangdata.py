# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import re
from tqdm import tqdm
import io
import sys
#æ”¹å˜æ ‡å‡†è¾“å‡ºçš„é»˜è®¤ç¼–ç ?
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

class XiaohuangData:
    """
    """

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.MAX_NUMBER_SUBDIR = 10
        self.conversations = []
        self.getLines()

    def getLines(self):
        #root_dir = "/home/tensor/chatbot/chatbot/corpus/xiaohuang"
        root_dir = "data/xiaohuang"
        qfile = open(root_dir + "/question.txt",'r', encoding="utf-8")
        afile = open(root_dir + "/answer.txt",'r', encoding="utf-8")
        i = 10
        q1 = ""
        a1 = ""

        for q in qfile:
            a = str(afile.readline())
            lines = []
            lines.append({"text": q})
            lines.append({"text": a})
            self.conversations.append({"lines": lines})

        qfile.close()
        afile.close()

    def getConversations(self):
        return self.conversations


#t = XiaohuangData("")
# #print(t.getConversations().pop())
