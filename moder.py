import asyncio
import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import re
import requests
import zipfile
import io
import os
import json
from datetime import datetime, timedelta
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.utils.data as data_utils
from transformers import pipeline
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import ChatPermissions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
L = logging.getLogger(__name__)

class R(Enum):
    U = "user"
    M = "moderator"
    A = "admin"
    O = "owner"

class V(Enum):
    T = "toxicity"
    S = "spam"
    F = "flood"
    L = "links"
    C = "caps"
    P = "political"

class C:
    T = ""
    B = 32
    E = 100
    M = 50
    V = 10000
    #for id tg-groups
    A = [

    ]
    #for name groups
    U = [
        # "skygaming_chat"
    ]
    
    W5 = 30
    W10 = 60 
    W15 = 3
    X = 24

class D:
    def __init__(self):
        self.c = set(C.A)
        self.u = set(C.U)

    def a(self, i: int, n: str = None) -> bool:
        if i in self.c:
            return True
        return False 

    def ac(self, i: int):
        self.c.add(i)
    def rc(self, i: int):
        self.c.discard(i)
    def au(self, u: str):
        self.u.add(u)
    def ru(self, u: str):
        self.u.discard(u)
    def gc(self):
        return list(self.c)
    def gu(self):
        return list(self.u)

class F:
    def __init__(self):

        self.p = {
            'хохол','укроп','укропитек','свидомит','бандера','майдаун',
            'украина','украинец','украинка', 'киев','львов','днепр',
            'гитлер','нацист','фашист','свастика','рейх','нацизм',
            'фашизм','гитлеризм','нигер',
            'z','zv','zov','вагнер','чвк','спецоперация',
            'националист','расист','шовинист','ксенофоб',}
        self.s = {
            'нигер': ['нига','nigers','niga','NIGA','Nigga','NiGa'],
            'хохол': ['хахол','хохлы','хахлы'],
            'укроп': ['укропы'],
            'бандера': ['бандеровец', 'бандеровцы'],
            'гитлер': ['гитлерюгенд'],
            'нацист': ['нацисты', 'нацистам'],
            'z': ['зет','zed','зэд','Z','ZOV','zOv','ZoV','ZOv','zOV','зов','ЗОв','зОВ','зОв','ЗоВ'],}
        
        self.f = set(self.p)
        for t, sl in self.s.items():
            self.f.update(sl)
    
    def c(self, t: str) -> bool:
        tl = t.lower(); w = re.findall(r'[a-zа-яё]+', tl)
        return any(word in self.f for word in w)

class U:
    def __init__(self):
        self.r = {}
        self.v = {}
        self.s = {}
        
    def sr(self, i: int, r: R):
        self.r[i] = r
    def gr(self, i: int) -> R:
        return self.r.get(i, R.U)
    def im(self, i: int) -> bool:
        r = self.gr(i)
        return r in [R.M, R.A, R.O]
    def ia(self, i: int) -> bool:
        r = self.gr(i)
        return r in [R.A, R.O]
    def av(self, i: int, t: V, s: float):
        if i not in self.v:
            self.v[i] = []
            
        v = {'type':t,'severity':s,'timestamp':datetime.now()}

        self.v[i].append(v)
        self._co(i)
        
    def gvc(self, i: int) -> int:
        return len(self.v.get(i, []))
    def grv(self, i: int, h: int = 24) -> list:
        if i not in self.v:
            return []

        ct = datetime.now() - timedelta(hours=h)
        return [v for v in self.v[i]if v['timestamp'] > ct]
    
    def _co(self, i: int):
        if i not in self.v:
            return
            
        ct = datetime.now() - timedelta(hours=C.X)
        self.v[i]=[v for v in self.v[i]if v['timestamp']>ct]

class P:
    def __init__(self):
        self.v = C.V
        self.m = C.M
        self.t = None
        
    async def i(self, texts):
        try:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            self.t = Tokenizer(num_words=self.v, oov_token="<OOV>")
            self.t.fit_on_texts(texts)
            L.info(f"Tokenizer initialized with {len(self.t.word_index)} words")
        except ImportError:
            L.warning("TensorFlow not available, using simple tokenizer")
            self._is(texts)
        
    def _is(self, texts):
        from collections import Counter
        w = []
        for t in texts:
            if isinstance(t,str):
                w.extend(self.p(t).split())
        
        wc = Counter(w); mc = wc.most_common(C.V-1)
        self.voc = {"<PAD>":0,"<OOV>":1}
        for i,(word, count)in enumerate(mc, start=2):
            self.voc[word]= i
        
        L.info(f"Simple tokenizer initialized with {len(self.voc)} words")
        
    def p(self, text):
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'http\S+|www\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def ts(self, texts):
        if self.t:
            s = self.t.texts_to_sequences(texts)
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            return pad_sequences(s, maxlen=self.m, padding='post', truncating='post')
        else:
            return self._sts(texts)
    
    def _sts(self, texts):
        s = []
        for t in texts:
            pt = self.p(t)
            w = pt.split()
            seq = [self.voc.get(word, 1) for word in w[:self.m]]
            seq = seq + [0] * (self.m - len(seq))
            s.append(seq)
        return np.array(s)

class N(nn.Module):
    def __init__(self, v=C.V, e=C.E, n=4, d=0.3):
        super().__init__()
        self.e = nn.Embedding(v, e, padding_idx=0)
        
        self.c = nn.ModuleList([
            nn.Conv1d(e, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
        ])
        
        self.f = nn.Sequential(
            nn.Linear(128, 64),nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(64, 32),nn.ReLU(),
            nn.Dropout(d),
            nn.Linear(32, n)
        )
        
    def forward(self, x):
        x = self.e(x);x = x.transpose(1, 2)
        
        for cl in self.c:
            x = torch.relu(cl(x));x = torch.max_pool1d(x, kernel_size=2)
            
        x = torch.max(x, dim=2)[0]
        return self.f(x)

class T:
    def __init__(self):
        self.w = set()
        self.cat = {'toxic':0,'severe_toxic':1,'obscene':2,'threat':3,'insult':4,'identity_hate':5,'clean':6,'ru_men':7,'ru_women':8}
        
    async def ld(self):
        try:
            if os.path.exists("train_data_version1.csv",""):
                df = pd.read_csv("train_data_version1.csv","")
                self._ew(df)
                L.info(f"Loaded {len(self.w)} toxic words from local dataset")
                return df
            else:
                L.warning("Local dataset not found, trying to download...")
                dc = await self._dd()
                if dc:
                    df = pd.read_csv(io.BytesIO(dc))
                    self._ew(df)
                    L.info(f"Loaded {len(self.w)} toxic words from downloaded dataset")
                    return df
                else:
                    raise Exception("No dataset available")
                
        except Exception as e:
            L.error(f"Dataset loading error: {e}")
            await self._lfd()
            return self._csd()
    
    async def _dd(self):
        urls = ["https://test.csv",]
        for u in urls:
            try:
                r = requests.get(u, timeout=30)
                if r.status_code == 200:
                    L.info(f"Successfully downloaded dataset from {u}")
                    return r.content
            except Exception as e:
                L.warning(f"Failed to download from {u}: {e}")
                continue
        return None
    
    def _ew(self, df):
        tc = df[
            (df['toxic'] == 1) | (df['severe_toxic'] == 1) | 
            (df['obscene'] == 1) | (df['threat'] == 1) |
            (df['insult'] == 1) | (df['identity_hate'] == 1)]['comment_text'].dropna()
        
        for c in tc:
            if isinstance(c, str):
                w = re.findall(r'\b[a-zа-я]+\b',c.lower());fw = [w for w in w if not self._ip(w)]
                self.w.update(fw)
    
    def _ip(self, w: str) -> bool:
        pr = {'еб','пизд','ху','бля','хер','гонд'}
        return any(r in w for r in pr)
    async def _lfd(self):
        ew = {'stupid','idiot','retard','moron','imbecile','jerk','scum','trash','garbage','loser','bastard'}
        rw = {'дурак','идиот','дебил','кретин','тупица','урод','шлюха','проститутка','ублюдок','падла','тварь','мразь','сволочь','мерзавец','негодяй','скотина'}
        
        self.w.update(ew)
        self.w.update(rw)
        L.info(f"Loaded fallback dictionary with {len(self.w)} words")
    
    def _csd(self):
        sc = ["This is a normal comment","You are stupid idiot","Hello world","I hate you all","Have a nice day","Go to hell bastard"] 
        return pd.DataFrame({'comment_text':sc})

class M:
    def __init__(self, model, dict, prep, pf):
        self.m = model
        self.d = dict
        self.p = prep
        self.pf = pf
        self.h = {}
        
    async def am(self, uid: int, msg: types.Message):
        tc = msg.text or ""
        

        f = await self._ef(tc);ms = await self._pt(tc);rs = await self._crs(uid,f,ms)
        await self._uh(uid,tc,f,ms,rs)
        return {'user_id':uid,'text':tc,'features':f,'ml_score':ms,'risk_score':rs,
        'risk_level':self._grl(rs),'timestamp':datetime.now()}
    
    async def _ef(self, text):
        if not text:
            return self._gef()
            
        tl = text.lower()
        w = re.findall(r'\b[a-zа-я]+\b', tl)
        tw = len(w)
        
        twf = [word for word in w if word in self.d.w]
        twc = len(twf)
        tr = twc / tw if tw > 0 else 0.0
        
        pc = self.pf.c(text)
        
        cl = bool(re.search(r'http|www|\.ru|\.com|\.net|t\.me|@\w+', tl))
        
        ss = await self._css(text)
        
        return {
            'len': len(text),
            'wc': tw,
            'cl': cl,
            'pc': pc,
            'cr': sum(1 for c in text if c.isupper()) / max(1, len(text)),
            'scr': len(re.findall(r'[!@#$%^&*()]', text)) / max(1, len(text)),
            'tr': min(1.0, tr * 2),
            'twc': twc,
            'tw': twf,
            'ss': ss
        }
    
    def _gef(self):
        return {
            'len': 0, 'wc': 0, 'cl': False,
            'pc': False, 'cr': 0.0, 
            'scr': 0.0, 'tr': 0.0,
            'twc': 0, 'tw': [], 'ss': 0.0
        }
    
    async def _css(self, text):
        sp = [
            (r'\b(бесплатно|даром|акция|скидка|распродажа)\b', 0.3),
            (r'\b(заработок|деньги|бизнес|доход|прибыль)\b', 0.2),
            (r'http|t\.me|@\w+|telegram', 0.4),
            (r'\b(купи|покупай|продам|закажи|оформи)\b', 0.2),
            (r'\b(срочно|горящее|успей|ограниченно)\b', 0.1)
        ]
        
        ss = 0.0
        for pat,pts in sp:
            if re.search(pat, text, re.IGNORECASE):
                ss += pts
                
        return min(1.0, ss)
    
    async def _pt(self, text):
        if not self.m:
            return 0.5
            
        try:
            pt = self.p.p(text);s = self.p.ts([pt]);ti = torch.LongTensor(s)
            
            with torch.no_grad():
                o = self.m(ti);prob = torch.softmax(o,dim=1);tp = prob[0,:6].sum().item()
                
            return tp
        except Exception as e:
            L.error(f"Prediction error: {e}")
            return 0.5
    
    async def _crs(self,uid,f,ms):
        w = {'toxicity':0.35,'spam':0.25,'ml_score':0.40}
        bs = (f['tr']* w['toxicity']+ f['ss']* w['spam']+ ms* w['ml_score'])
        
        if f['cl']:bs+=1.0
        if f['pc']:bs+=0.8
        
        if uid in self.h and self.h[uid]:
            rs = []
            for e in self.h[uid][-5:]:
                if 'risk_score' in e:
                    rs.append(e['risk_score'])
            
            if rs:
                hi = np.mean(rs)*0.2; bs += hi
                
        return min(1.0,bs)
    
    def _grl(self, s):
        if s >= 0.8:
            return "CRITICAL"
        elif s >= 0.6:
            return "HIGH"
        elif s >= 0.4:
            return "MEDIUM"
        elif s >= 0.2:
            return "LOW"
        else:
            return "NORMAL"
    
    async def _uh(self,uid,text,f,ms,rs):
        if uid not in self.h: self.h[uid]=[]
            
        self.h[uid].append({'text':text,'features':f,'ml_score':ms,'risk_score':rs,
            'timestamp':datetime.now()})
    
        if len(self.h[uid])>50:
            self.h[uid] = self.h[uid][-50:]

class B:
    def __init__(self,token:str):
        self.b = Bot(token=token)
        self.d = Dispatcher(storage=MemoryStorage())
        self.cm = D()
        self.pp = P()
        self.td = T()
        self.m = N()
        self.pf = F()
        self.ma = M(self.m,self.td,self.pp,self.pf)
        self.um = U()
        self.us = {}
        self._rh()
    
    async def i(self):
        L.info("Initializing...")
        
        ds = await self.td.ld()
        if ds is not None:
            texts = ds['comment_text'].fillna('').tolist()
            await self.pp.i(texts)
            L.info("Models initialized with dataset")
        else:
            L.warning("Using pre-trained weights")
            
        L.info(f"Toxic dictionary contains {len(self.td.w)} words")
        L.info(f"Political filter: {len(self.pf.f)} terms")
    
    async def _cca(self, msg: types.Message) -> bool:
        cu = getattr(msg.chat, 'username', None)

        if not self.cm.a(msg.chat.id):
            if msg.chat.type in ['group', 'supergroup']:
                await msg.answer("❌ Этот бот не разрешён для использования в этом чате.")
            return False
        return True
    
    def _rh(self):
        @self.d.message(Command("start"))
        async def sc(msg: types.Message):
            if not await self._cca(msg):
                return
                
            wt = (
                "Sky Gaming AI-moderator\n\n"
                "Автоматическая модерация:\n"
                "• Запрет политических тем\n"
                "• Блокировка ссылок\n"
                "• Контроль оскорблений\n"
                "• Авто-мут за нарушения\n\n"
                "📊 Команды:\n"
                "/stats - статистика\n"
                "/userinfo - информация о пользователе\n\n"
                "Бот работает автоматически!"
            )
            await msg.answer(wt)
        
        @self.d.message(Command("stats"))
        async def stc(msg: types.Message):
            if not await self._cca(msg):
                return
                
            tu = len(self.us)
            hru = len([s for s in self.us.values() if s > 0.6])
            mods = len([uid for uid, role in self.um.r.items() 
                             if self.um.im(uid)])
            
            await msg.answer (
                "STATISTICS\n\n"
                f"• Всего пользователей: {tu}\n"
                f"• Пользователей высокого риска: {hru}\n"
                f"• Модераторов: {mods}\n"
                f"• Токсичный словарь: {len(self.td.w)} слов\n"
                f"• Политических терминов: {len(self.pf.f)}\n"
                f"• Статус: 🟢 АКТИВЕН"
            )
        
        @self.d.message(Command("userinfo"))
        async def uic(msg: types.Message):
            if not await self._cca(msg):
                return
                
            if not msg.reply_to_message:
                await msg.answer("❌ Ответьте на сообщение пользователя")
                return
                
            uid = msg.reply_to_message.from_user.id
            role = self.um.gr(uid); viol = self.um.gvc(uid); rs = self.us.get(uid, 0)
            
            ui = (f"👤 ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ\n\n" 
            f"• Роль: {role.value}\n • Risk Score: {rs:.1%}\n • Нарушения: {viol}\n • Статус: {'🟢 Норма'if rs<0.5 else'🟡 Наблюдение'if rs<0.8 else'🔴 Высокий риск'}")
            await msg.answer(ui)
        
        @self.d.message()
        async def aam(msg: types.Message):
            if not await self._cca(msg):return  
            if not msg.text or msg.text.startswith('/'):return
            if self.um.im(msg.from_user.id):return
                
            ar = await self.ma.am(msg.from_user.id, msg)
            
            uid = msg.from_user.id; cs = self.us.get(uid, 0); ns = cs + ar['risk_score'] * 0.1; self.us[uid] = min(1.0, ns)
            
            await self._ham(msg, ar)
    
    async def _ham(self, msg: types.Message, an):
        uid = msg.from_user.id
        f = an['features']
        
        try:
            if f['cl']:
                await msg.delete()
                await msg.answer(
                    f"🚫 Ссылки запрещены в этом чате!\n"
                    f"Сообщение от @{msg.from_user.username or 'пользователя'} удалено."
                )
                self.um.av(uid, V.L, 1.0)
                return
            
            if f['pc']:
                await msg.delete()
                await msg.answer(
                    f"🚫 Политические обсуждения запрещены!\n"
                    f"Сообщение от @{msg.from_user.username or 'пользователя'} удалено."
                )
                self.um.av(uid, V.P, 1.0)
                return
            
            tc = f['twc']
            if tc >= 15:
                ud = datetime.now() + timedelta(days=C.W15)
                await self._am(msg, uid, ud, f"15+ оскорблений ({tc} слов)")
                self.um.av(uid, V.T, 1.0)
                
            elif tc >= 10:
                ud = datetime.now() + timedelta(minutes=C.W10)
                await self._am(msg, uid, ud, f"10+ оскорблений ({tc} слов)")
                self.um.av(uid, V.T, 0.8)
                
            elif tc >= 5:
                ud = datetime.now() + timedelta(minutes=C.W5)
                await self._am(msg, uid, ud, f"5+ оскорблений ({tc} слов)")
                self.um.av(uid, V.T, 0.6)
                
            elif tc > 0:
                await msg.reply(
                    f"⚠️ Обнаружены оскорбления: {', '.join(f['tw'][:3])}\n"
                    f"Следующее нарушение - мут."
                )
                self.um.av(uid, V.T, 0.3)
                
            elif an['risk_score'] > 0.7:
                self.um.av(uid, V.T, an['risk_score'])
                
                vc = self.um.gvc(uid)
                
                if an['risk_score'] > 0.9 and vc >= 5:
                    await msg.delete()
                    await self.b.ban_chat_member(chat_id=msg.chat.id, user_id=uid)
                    await msg.answer(f"🚫 Пользователь забанен за повторные нарушения")
                    
                elif an['risk_score'] > 0.8 and vc >= 3:
                    await msg.delete()
                    ud = datetime.now() + timedelta(minutes=60)
                    await self.b.restrict_chat_member(
                        chat_id=msg.chat.id, user_id=uid,
                        permissions=ChatPermissions(can_send_messages=False), until_date=ud
                    )
                    await msg.answer(f"🔇 Пользователь замучен на 60 минут")
                    
                elif an['risk_score']>0.8:
                    await msg.delete()
                    await msg.answer("⚠️ Сообщение удалено за высокий риск")
                elif an['risk_score']>0.7:
                    await msg.reply("⚠️ Обнаружен контент высокого риска")
                    
        except Exception as e:
            L.error(f"Auto-moderation failed: {e}")
    
    async def _am(self, msg: types.Message, uid: int, ud: datetime, r: str):
        try:
            await self.b.restrict_chat_member(chat_id=msg.chat.id,user_id=uid,permissions=ChatPermissions(can_send_messages=False),until_date=ud)
            
            d = self._fd(ud - datetime.now())
            await msg.answer(
                f"🔇 Пользователь @{msg.from_user.username or 'замучен'} "
                f"получил мут на {d}\n"
                f"Причина: {r}"
            )
            
        except Exception as e:
            L.error(f"Mute failed: {e}")
            await msg.answer("Не удалось выдать мут. Проверьте права бота.")
    
    def _fd(self, d: timedelta) -> str:
        if d.days > 0:
            return f"{d.days} дней"
        elif d.seconds >= 3600:
            h = d.seconds // 3600
            return f"{h} часов"
        else:
            m = d.seconds // 60
            return f"{m} минут"

async def main():
    if C.T == "YOUR_BOT_TOKEN_HERE":
        L.error("Please replace BOT_TOKEN in configuration")
        return
        
    try:
        bi = B(C.T)
        L.info("Starting AI Moderator...")
        L.info("Initializing models...")
        
        await bi.i()
        L.info("Models ready")
        L.info("Bot is running with full moderation capabilities")
        
        await bi.d.start_polling(bi.b)
    except Exception as e:
        L.error(f"Startup error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
