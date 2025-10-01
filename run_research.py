# run_research.py
# 目的:
# - Deep Research を使い、ヘッジファンドPM視点の銘柄レポートをMarkdown出力
# - 6210 から毎実行1銘柄ずつ前進（GITHUB_RUN_NUMBER で管理）
# - o3 が未解放の場合は o4-mini に自動フォールバック

from openai import OpenAI
import openai  # 例外クラス用
from zoneinfo import ZoneInfo
import os, datetime, pathlib, re

# ====== 設定 ======
PRIMARY_MODEL = os.environ.get("DR_MODEL", "o3-deep-research")  # 高品質を試す
FALLBACK_MODEL = "o4-mini-deep-research"                         # 未認証でも使いやすい
BASE_CODE = int(os.environ.get("DR_BASE_CODE", "6210"))
OUTDIR = pathlib.Path(os.environ.get("DR_OUTDIR", "outputs"))
OUTDIR.mkdir(exist_ok=True)

run_no = int(os.environ.get("GITHUB_RUN_NUMBER", "1"))
target_code = BASE_CODE + (run_no - 1)

now_jst = datetime.datetime.now(ZoneInfo("Asia/Tokyo"))
ts = now_jst.strftime("%Y%m%d-%H%M")

def slot_label(dt: datetime.datetime) -> str:
    h = dt.hour
    if h < 10: return "Morning-08JST"
    if h < 14: return "Midday-12JST"
    if h < 19: return "Evening-17JST"
    return "Night-21JST"
slot = slot_label(now_jst)

# ====== プロンプト ======
SYSTEM_MESSAGE = (
    "あなたは機関投資家（ヘッジファンド）のPM兼リサーチ責任者です。"
    "厳密な出典と最新データに基づき、日本語で、Markdown構造のエクイティ・リサーチレポートを作成します。"
    "曖昧な主張は避け、必ず根拠リンク（注釈）を本文に残してください。"
    "法令・規制を尊重し、末尾に投資勧誘ではない旨のディスクレーマーを含めてください。"
)

USER_QUERY = f"""
以下の『日本の証券コード』に対応する上場企業について、ヘッジファンドPM視点の
アナリストレポートを作成してください。レポートは **1銘柄のみ** 対象。

- 対象コード: **{target_code}**（東京証券取引所、4桁コード）
- 生成日時（JST）: {now_jst:%Y-%m-%d %H:%M} ({slot})

## 出力要件（Markdown）
# <企業正式名>（<証券コード>）— ヘッジファンド銘柄レポート
- 企業名（日/英）、証券コード、上場市場、ティッカー表記（例: TSE:{target_code}）
- 株価・時価総額・出来高（可能なら最新営業日ベース。難しければ注釈付きでNA）
- データ/ニュースの**日付は YYYY-MM-DD で明記**
- **冒頭に5行以内のエグゼクティブサマリー**

## 1. 会社概要
## 2. 財務サマリー（3〜5年推移・四半期要点・主要指標）
## 3. バリュエーション（PER/PBR/EV/EBITDA/配当、ピア比較）
## 4. 戦略・競争優位（中計/投資/技術/サプライ/規制）
## 5. 直近ニュース/イベント（TDnet/IR/報道：日付・要約・投資インパクト）
## 6. リスク（需要/価格/為替/供給/規制/財務）
## 7. 投資シナリオ（Bull/Base/Bear：前提・レンジ・トリガー）
## 8. スタンス（Long/Short/Watch、期間、コンビクション1〜5）
## 9. 参考資料（本文注釈から自動生成）

### 重要
- コードから正式社名と市場名を同定。数値は出典と時点を明記。公開情報のみ。
"""

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def create_response(model_name: str):
    return client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [{"type": "input_text", "text": USER_QUERY}]},
        ],
        tools=[{"type": "web_search_preview"}],  # Deep Researchはデータソース必須
        # reasoning は使用しない（未認証Orgだとエラーになる場合がある）
    )

# まずは PRIMARY_MODEL（例: o3）で試行 → ダメなら o4-mini にフォールバック
model_used = PRIMARY_MODEL
try:
    resp = create_response(PRIMARY_MODEL)
except (openai.NotFoundError, openai.BadRequestError) as e:
    msg = str(e).lower()
    # よくある条件：model_not_found / verify organization / not allowed
    if "model_not_found" in msg or "verify organization" in msg or "not allowed" in msg:
        model_used = FALLBACK_MODEL
        print(f"[WARN] {PRIMARY_MODEL} が使用不可のため {FALLBACK_MODEL} にフォールバックします。詳細: {e}")
        resp = create_response(FALLBACK_MODEL)
    else:
        raise

# ====== 本文と注釈の抽出 ======
def extract_text_and_annotations(r):
    try:
        content = r.output[-1].content[0]
        text = getattr(content, "text", "") or ""
        annotations = getattr(content, "annotations", []) or []
        return text, annotations
    except Exception:
        try:
            return getattr(r, "output_text", ""), []
        except Exception:
            return "", []

def ann_get(ann, key, default=""):
    if isinstance(ann, dict):
        return ann.get(key, default)
    return getattr(ann, key, default)

report_text, annotations = extract_text_and_annotations(resp)

# 参考文献の整形
def normalize_link(url: str) -> str:
    return re.sub(r"(\?|\&)(utm_[^=]+|_hsenc|_hsmi)=[^&]+", "", url or "")

bib_lines = []
for i, ann in enumerate(annotations, start=1):
    title = (ann_get(ann, "title") or "source").strip()
    url = normalize_link(ann_get(ann, "url"))
    if url:
        bib_lines.append(f"{i}. [{title}]({url})")

disclaimer = (
    "\n\n---\n"
    "**ディスクレーマー**: 本資料は情報提供のみを目的としており、特定の有価証券の売買の勧誘、"
    "投資助言、または保証を構成するものではありません。投資判断は自己責任で行ってください。"
)

full_md = report_text
if bib_lines:
    full_md += "\n\n## References\n" + "\n".join(bib_lines)
full_md += disclaimer + f"\n\n*Model used: **{model_used}***"

outfile = OUTDIR / f"equity_report_{target_code}_{ts}.md"
outfile.write_text(full_md, encoding="utf-8")
print(f"[OK] Saved: {outfile}  (Run #{run_no}, Code {target_code}, JST {now_jst:%Y-%m-%d %H:%M}, Model {model_used})")
