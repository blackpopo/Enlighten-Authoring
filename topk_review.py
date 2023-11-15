from streamlit_utils import *
from utils import *

def generate_topk_review_component():
    if 'papers_df' in st.session_state:
        display_spaces(2)
        display_description("AI レビュー生成", 3)

        st.session_state['number_of_review_papers'] = st.slider(
            f"レビューに使用する論文数を選択してください。",   min_value=1, value=20,  max_value=min(100, len(st.session_state['papers_df'])), step=1)

        toggle = display_language_toggle(f'レビュー生成')
        st.session_state['topk_review_toggle'] = toggle

        topk_review_button = st.button(f"上位 {st.session_state['number_of_review_papers']} 件の論文によるレビュー生成。(時間がかかります)",)
        if topk_review_button:
            with st.spinner("⏳ AIによるレビューの生成中です。 お待ち下さい..."):
                response, links, caption, draft_references = title_review_papers(st.session_state['papers_df'][:st.session_state['number_of_review_papers']], st.session_state['query'], model = 'gpt-4-32k', language=toggle)
                st.session_state['topk_review_caption'] = "上位" + caption
                st.session_state['topk_review_response'] = response

                reference_indices = extract_reference_indices(response)
                references_list = [reference_text for i, reference_text in enumerate(links) if i in reference_indices]
                draft_references_list = [reference_text for i, reference_text in enumerate(draft_references) if i in reference_indices]
                st.session_state['topk_references_list'] = references_list
                st.session_state['topk_draft_references_list'] = draft_references_list
def display_topk_review_component():
    #レビュー内容の常時表示
    if 'papers_df' in st.session_state and 'topk_review_response' in st.session_state and 'topk_review_caption' in st.session_state and 'topk_references_list' in st.session_state:
        display_description(st.session_state['topk_review_caption'], size=5)
        display_spaces(1)
        display_list(st.session_state['topk_review_response'].replace('#', '').split('\n'), size=8)

        if len(st.session_state['topk_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['topk_references_list'])

def generate_next_topk_review_component():
    if 'papers_df' in st.session_state and 'topk_review_response'in st.session_state and 'topk_review_caption' in st.session_state and 'topk_references_list' in st.session_state:
        next_topk_review_button = st.button(f"次の上位 {st.session_state['number_of_review_papers']} 件の論文によるレビュー生成。(時間がかかります)")
        if next_topk_review_button:
            if not 'next_number_of_review_papers' in st.session_state:
                st.session_state['next_number_of_review_papers'] = st.session_state['number_of_review_papers'] * 2
            elif ('next_number_of_review_papers' in st.session_state) and (
                    st.session_state['next_number_of_review_papers'] < len(st.session_state['papers_df'])):
                st.session_state['next_number_of_review_papers'] = st.session_state['number_of_review_papers'] + \
                                                                   st.session_state['next_number_of_review_papers']
            else:
                st.session_state['next_number_of_review_papers'] = st.session_state['number_of_review_papers']

            button_title = f"上位 {st.session_state['next_number_of_review_papers'] - st.session_state['number_of_review_papers'] + 1} 件目から {min(st.session_state['next_number_of_review_papers'], len(st.session_state['papers_df']))} 件目"

            with st.spinner(f"⏳ {button_title} の論文を使用した AI によるレビューの生成中です。 お待ち下さい..."):
                response, links, caption, draft_references = title_review_papers(
                    st.session_state['papers_df'][st.session_state['next_number_of_review_papers'] - st.session_state['number_of_review_papers']: st.session_state['next_number_of_review_papers']],
                    st.session_state['query'], model='gpt-4-32k', language=st.session_state['topk_review_toggle'] )
                st.session_state['topk_review_caption'] = f"{button_title}の論文による" + caption
                st.session_state['topk_review_response'] = response

                reference_indices = extract_reference_indices(response)
                references_list = [reference_text for i, reference_text in enumerate(links) if i in reference_indices]
                draft_references_list = [reference_text for i, reference_text in enumerate(draft_references) if i in reference_indices]
                st.session_state['topk_references_list'] = references_list
                st.session_state['topk_draft_references_list'] = draft_references_list
                st.experimental_rerun()
def generate_topk_draft_component():
    if 'papers_df' in st.session_state:
        display_spaces(1)
        display_description(f"文章の草稿を入力してください。上位 {st.session_state['number_of_review_papers']} 件のレビューによりエビデンスを付与します。", 5)
        #ドラフトの入力部分
        if not 'topk_draft_text' in st.session_state:
            draft_text = st.text_area(label='review draft input filed.', placeholder='Past your draft of review here.', label_visibility='hidden', height=300)
        else:
            draft_text = st.text_area(label='review draft input filed.', value = st.session_state['topk_draft_text'],placeholder='Past your draft of review here.', label_visibility='hidden', height=300)
        st.session_state['topk_draft_text'] = draft_text

        toggle = display_language_toggle(f"レビューによるエビデンス付与")
        mode_toggle = display_draft_evidence_toggle(f"レビュー生成")

        write_summary_button = st.button(f"上位 {st.session_state['number_of_review_papers']} 件の論文レビューによるエビデンス付与。(時間がかかります)", )

        if write_summary_button and len(draft_text) > 0:
            if 'topk_review_response' in st.session_state and 'topk_draft_references_list' in st.session_state:
                with st.spinner("⏳ AIによるエビデンスの付与中です。 お待ち下さい..."):
                    topk_summary_response, caption = summery_writer_with_draft(st.session_state['topk_review_response'], draft_text, st.session_state['topk_draft_references_list'], model = 'gpt-4-32k', language=toggle, mode=mode_toggle)
                    display_description(caption)
                    display_spaces(2)

                    st.session_state['topk_summary_response'] = topk_summary_response

                    reference_indices = extract_reference_indices(topk_summary_response)
                    references_list = [reference_text for i, reference_text in enumerate(st.session_state['topk_references_list']) if
                                       i in reference_indices]
                    st.session_state['topk_summary_references_list'] = references_list
            else:
                display_description("レビューがありません。先にレビューを生成してください。")
        elif write_summary_button:
            display_description("入力欄が空白です。草稿を入力してください。")
def display_topk_draft_component():
    if 'topk_summary_response' in st.session_state and 'topk_summary_references_list' in st.session_state:
        display_list(st.session_state['topk_summary_response'].replace('#', '').split('\n'), size=8)

        if len(st.session_state['topk_summary_references_list']) > 0:
            display_description('参考文献リスト', size=6)
            display_references_list(st.session_state['topk_summary_references_list'])


def topk_review_papers():
    #すでに papers のデータフレームがあれば、それを表示する。
    if 'papers_df' in st.session_state:
        display_dataframe_detail(st.session_state['papers_df'],  f'論文検索結果上位 20 件', 20)
    # クラスターのレビュー生成

    generate_topk_review_component()

    display_topk_review_component()

    generate_next_topk_review_component()

    display_spaces(2)

    # クラスターの草稿編集

    generate_topk_draft_component()

    display_topk_draft_component()
