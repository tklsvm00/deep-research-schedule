# run_research.py
# 目的:
# - Deep Research (高品質) を使い、ヘッジファンドPM視点の銘柄分析レポートをMarkdownで出力
# - 証券コード 6210 から開始し、実行ごとに 1 銘柄ずつ順番に進める
# - シーケンスは GitHub Actions の GITHUB_RUN_NUMBER を利用（初回=6210, 2回目=6211, ...）
# - 4回/日の実行はワークフロー(cron)側で設定してください（本スクリプトは「1回の起動=1銘柄」）

from openai import OpenAI
from zoneinfo import ZoneInfo
import os, datetime, pathlib, re

# ====== 設定 ======
MODEL = os.environ.get("DR_MODEL", "o3-deep-research")  # 高品質モード
BASE_CODE = int(os.environ.get("DR_BASE_CODE", "6210"))  # 開始コード（デフォルト: 6210）
OUTDIR = pathlib.Path(os.environ.get("DR_OUTDIR", "outputs"))
OUTDIR.mkdir(exist_ok=True)

# GitHub Actionsの連番（なければ 1 とみなす）
run_no = int(os.environ.get("GITHUB_RUN_NUMBER", "1"))
target_code = BASE_CODE + (run_no - 1)

# 日本時間タイムスタンプ（レポートの見出しやファイル名に使用）
now_jst = datetime.datetime.now(ZoneInfo("Asia/Tokyo"))
ts = now_jst.strftime("%Y%m%d-%H%M")

# 実行スロットの表示（任意：8/12/17/21時実行を識別するためのラベル）
def slot_label(dt: datetime.datetime) -> str:
    h = dt.hour
    if h < 10: return "Morning-08JST"
    if h < 14: return "Midday-12JST"
    if h < 19: return "Evening-17JST"
    return "Night-21JST"
slot = slot_label(now_jst)

# ====== プロンプト ======
SYSTEM_MESSAGE = (
    "あなたは機関投資家（ヘッジファンド）のポートフォリオマネージャー兼リサーチ責任者です。"
    "厳密な出典と最新データに基づき、日本語で、Markdown構造のエクイティ・リサーチレポートを作成します。"
    "曖昧な主張は避け、必ず根拠リンク（注釈）を本文に残してください。"
    "法令・規制を尊重し、レポート末尾に『本資料は情報提供目的であり投資勧誘ではない』旨のディスクレーマーを含めてください。"
)

# 会社名はモデル側でコードから特定させる（上場区分や市場名も）
USER_QUERY = f"""
以下の『日本の証券コード』に対応する上場企業について、ヘッジファンドPM視点の
アナリストレポートを作成してください。レポートは **1銘柄のみ** 対象とします。

- 対象コード: **{target_code}**（東京証券取引所、4桁コード）
- 生成日時（JST）: {now_jst:%Y-%m-%d %H:%M} ({slot})

## 出力要件（Markdown）
# <企業正式名>（<証券コード>）— ヘッジファンド銘柄レポート
- 企業名（日本語/英語）、証券コード、上場市場、ティッカー表記（例: TSE:{target_code}）
- 株価・時価総額・出来高（可能なら最新の営業日ベース）※取得困難時は注釈付きでNA可
- 参考日付：データ時点・ニュース日付は**YYYY-MM-DD**で明記

## 1. 会社概要
- 事業セグメント、主要製品/サービス、顧客、地理的内訳、沿革（重要なM&A/再編）
- 経営陣（主要人物と役割）、株主構成（主要株主・浮動株の概観）

## 2. 財務サマリー（単位と通貨を明記）
- 直近3〜5年の売上・営業利益・営業CF・FCF・ROE・ROIC（表形式）
- 粗利/営業利益率/在庫回転/設備投資 等の主要指標（可能な範囲で）
- 直近四半期ハイライト（決算短信/補足資料/IRから要点）

## 3. バリュエーション
- PER / PBR / EV/EBITDA / 配当利回り（可能ならTTMまたはFYベースを明記）
- 国内/海外 同業比較（ピアのコードや社名を付記）。差異の要因分析

## 4. 戦略・競争優位
- 中期経営計画や成長ドライバー（新製品/設備投資/地理展開/価格改定）
- 技術・サプライチェーン・規制面のモート/参入障壁

## 5. 直近ニュース / イベント
- 重要開示（TDnet）、IRリリース、日経/各社ニュース（**日付+要約+投資インパクト**）
- カタリスト（決算/ガイダンス/新製品/規制/大型受注 等）

## 6. リスク
- 需要/価格/為替/サプライ/規制/財務の主要リスクとモニタリング指標

## 7. 投資シナリオ（数値仮定は保守/合理的な範囲で）
- **Bull / Base / Bear** の3シナリオ（主要前提・売上/利益レンジ、トリガー）
- シナリオ確度の定性的評価とウォッチ項目

## 8. スタンス（情報目的）
- スタンス: **Long / Short / Watch** のいずれか（根拠を簡潔に）
- 想定投資期間: 3〜6ヶ月 / 12ヶ月
- コンビクション: 1〜5段階（根拠の質と一貫性を明示）

## 9. 参考資料（本文の注釈から自動的に生成）
- TDnet、EDINET、有価証券報告書、決算短信、会社IR、証券会社レポ要旨、報道 等

### 重要
- **会社名・市場名** をコードから正しく同定し、正式表記を使用。
- 数値は**出典と時点**を記載。取得不可は NA とし、代替理由を注釈に。
- 機密情報は用いない。Web上の公開情報に限定。
"""

# ====== API 呼び出し ======
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

resp = client.responses.create(
    model=MODEL,
    input=[
        {"role": "developer", "content": [{"type": "input_text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "input_text", "text": USER_QUERY}]},
    ],
    tools=[{"type": "web_search_preview"}],  # Deep Researchはデータソースが必須
    reasoning={"summary": "auto"},
)

# ====== 本文と注釈の抽出（冗長防止のため堅めにパース） ======
def extract_text_and_annotations(r):
    try:
        content = r.output[-1].content[0]
        text = getattr(content, "text", "") or ""
        annotations = getattr(content, "annotations", []) or []
        return text, annotations
    except Exception:
        # フォールバック（構造が変わった場合）
        try:
            return getattr(r, "output_text", ""), []
        except Exception:
            return "", []

report_text, annotations = extract_text_and_annotations(resp)

# 参考文献（URL）の抽出
def normalize_link(url: str) -> str:
    # 余計なトラッキング等を簡易除去（最低限）
    return re.sub(r"(\?|\&)(utm_[^=]+|_hsenc|_hsmi)=[^&]+", "", url)

bib_lines = []
for i, ann in enumerate(annotations, start=1):
    title = (getattr(ann, "title", "") or "source").strip()
    url = normalize_link(getattr(ann, "url", "") or "")
    if url:
        bib_lines.append(f"{i}. [{title}]({url})")

# 末尾にReferencesとディスクレーマーを付与
disclaimer = (
    "\n\n---\n"
    "**ディスクレーマー**: 本資料は情報提供のみを目的としており、特定の有価証券の売買の勧誘、投資助言、"
    "または保証を構成するものではありません。投資判断は自己責任で行ってください。"
)

full_md = report_text
if bib_lines:
    full_md += "\n\n## References\n" + "\n".join(bib_lines)
full_md += disclaimer

# ====== 保存 ======
outfile = OUTDIR / f"equity_report_{target_code}_{ts}.md"
outfile.write_text(full_md, encoding="utf-8")

print(f"[OK] Saved: {outfile}  (Run #{run_no}, Code {target_code}, JST {now_jst:%Y-%m-%d %H:%M})")
