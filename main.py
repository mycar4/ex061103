import streamlit as st
import tempfile #폴더를 임시로 만든다
import os
import json
import pandas as pd
from dotenv import load_dotenv

# --- 최신 LangChain 패키지 구조 Import ---
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 1. 환경 변수 로드 ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- 2. UI 글로벌 세팅 ---
st.set_page_config(page_title="사내 AI 스마트 워크스페이스", page_icon="🚀", layout="wide")

if not openai_api_key:
    st.error("🚨 `.env` 파일에서 OPENAI_API_KEY를 찾을 수 없습니다. 설정을 확인해주세요.")
    st.stop()

# --- 3. 라우팅용 글로벌 세션 상태 초기화 ---
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "🏠 메인 포탈"

# --- 4. 사이드바 글로벌 내비게이션 ---
with st.sidebar:
    st.title("🌐 스마트 워크스페이스")
    st.caption("AI 에이전트 통합 포탈")
    st.divider()
    
    # 사이드바 퀵 메뉴 (어디서나 홈으로 가거나 다른 앱으로 점프 가능)
    menu_options = ["🏠 메인 포탈", "📚 사내 지식 Q&A 시스템", "🎧 유튜브 요약 & Q&A", "⚖️ 법률/의료 조항 분석기"]
    current_idx = menu_options.index(st.session_state.app_mode)
    
    selected_mode = st.selectbox("🚀 신속 이동 메뉴", menu_options, index=current_idx)
    if selected_mode != st.session_state.app_mode:
        st.session_state.app_mode = selected_mode
        st.rerun()

# =========================================================================
# 🏠 CASE 0: 웰컴 게이트 페이지 (메인 대시보드)
# =========================================================================
if st.session_state.app_mode == "🏠 메인 포탈":
    st.title("🚀 사내 AI 스마트 워크스페이스 포탈")
    st.markdown("#### 업무 효율을 극대화하기 위한 맞춤형 AI 에이전트 툴킷입니다. 필요한 도구를 선택하세요.")
    st.write("")
    st.write("")
    
    # 3열 카드 레이아웃 구성
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border-left: 5px solid #0d6efd; min-height: 220px;">
                <h3>📚 사내 지식 Q&A</h3>
                <p>회사 규정, 가이드라인, 매뉴얼 PDF 등 사내 문서를 학습시켜 보안 걱정 없이 질문하고 답변을 받아보세요.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button("Q&A 시스템 진입 👉", key="go_kb", use_container_width=True):
            st.session_state.app_mode = "📚 사내 지식 Q&A 시스템"
            st.rerun()
            
    with col2:
        st.markdown(
            """
            <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border-left: 5px solid #ffc107; min-height: 220px;">
                <h3>🎧 유튜브 요약 & Q&A</h3>
                <p>영어 등 외국어 영상도 한글로 자동 번역! AI가 자막을 분석하여 핵심 요약 및 대화형 질의응답을 지원합니다.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button("유튜브 요약 진입 👉", key="go_yt", use_container_width=True):
            st.session_state.app_mode = "🎧 유튜브 요약 & Q&A"
            st.rerun()
            
    with col3:
        st.markdown(
            """
            <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border-left: 5px solid #dc3545; min-height: 220px;">
                <h3>⚖️ 법률/의료 조항 분석기</h3>
                <p>복잡한 계약서 조항이나 특수 의료 가이드라인 텍스트를 파싱하여 독소조항, 의무사항, 고위험군 조항을 구조화하여 시각화합니다.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button("조항 분석기 진입 👉", key="go_az", use_container_width=True):
            st.session_state.app_mode = "⚖️ 법률/의료 조항 분석기"
            st.rerun()

# =========================================================================
# 📚 CASE 1: 사내 지식 기반 Q&A 시스템
# =========================================================================
elif st.session_state.app_mode == "📚 사내 지식 Q&A 시스템":
    c_head, c_home = st.columns([5, 1])
    with c_head:
        st.header("📚 사내 지식 기반 Q&A 챗봇")
    with c_home:
        if st.button("🏠 포탈로 복귀", use_container_width=True, key="btn_home_kb"):
            st.session_state.app_mode = "🏠 메인 포탈"
            st.rerun()
            
    st.caption("사내 문서(PDF)를 업로드하고 질문하면, AI가 문서 내용을 바탕으로 답변합니다.")

    if "kb_messages" not in st.session_state: st.session_state.kb_messages = []
    if "kb_vector_store" not in st.session_state: st.session_state.kb_vector_store = None
    if "kb_filename" not in st.session_state: st.session_state.kb_filename = ""

    with st.sidebar:
        st.subheader("사내 문서 업로드")
        kb_file = st.file_uploader("PDF 파일을 업로드하세요.", type="pdf", key="kb_upload")

    if kb_file and st.session_state.kb_filename != kb_file.name:
        st.session_state.kb_messages = [] 
        with st.spinner(f"'{kb_file.name}' 분석 및 벡터 DB 구축 중..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(kb_file.getvalue())
                    tmp_file_path = tmp_file.name
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                os.remove(tmp_file_path)

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)

                embeddings = OpenAIEmbeddings()
                st.session_state.kb_vector_store = InMemoryVectorStore.from_documents(splits, embeddings)
                st.session_state.kb_filename = kb_file.name
                st.success("✅ 문서 학습 완료!")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    for msg in st.session_state.kb_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if kb_prompt := st.chat_input("업로드한 사내 문서에 대해 질문해주세요.", key="kb_input"):
        if st.session_state.kb_vector_store is None:
            st.warning("👈 먼저 사이드바에서 PDF 문서를 업로드해주세요.")
            st.stop()

        st.session_state.kb_messages.append({"role": "user", "content": kb_prompt})
        with st.chat_message("user"): st.markdown(kb_prompt)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        sys_prompt = "반드시 아래 문맥(Context)만을 기반으로 답변하세요. 모르면 모른다고 답하세요.\n\nContext:\n{context}"
        prompt_template = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{input}")])

        retriever = st.session_state.kb_vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        with st.chat_message("assistant"):
            with st.spinner("문서 검색 중..."):
                res = rag_chain.invoke({"input": kb_prompt})
                st.markdown(res["answer"])
                with st.expander("참고한 문서 출처"):
                    for i, doc in enumerate(res["context"]):
                        st.markdown(f"**Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')})**")
                        st.text(doc.page_content[:200] + "...")
        st.session_state.kb_messages.append({"role": "assistant", "content": res["answer"]})

# =========================================================================
# 🎧 CASE 2: 유튜브 영상/팟캐스트 요약 및 Q&A (한글 번역 로직 대폭 강화)
# =========================================================================
elif st.session_state.app_mode == "🎧 유튜브 요약 & Q&A":
    c_head, c_home = st.columns([5, 1])
    with c_head:
        st.header("🎧 유튜브 영상 요약 및 Q&A 챗봇")
    with c_home:
        if st.button("🏠 포탈로 복귀", use_container_width=True, key="btn_home_yt"):
            st.session_state.app_mode = "🏠 메인 포탈"
            st.rerun()
            
    st.caption("유튜브 URL을 입력하면 영상 언어에 관계없이 자동으로 한글 번역 요약 및 한국어 질의응답을 제공합니다.")

    if "yt_messages" not in st.session_state: st.session_state.yt_messages = []
    if "yt_vector_store" not in st.session_state: st.session_state.yt_vector_store = None
    if "yt_summary" not in st.session_state: st.session_state.yt_summary = ""
    if "yt_url" not in st.session_state: st.session_state.yt_url = ""

    with st.sidebar:
        st.subheader("유튜브 링크 입력")
        yt_url_input = st.text_input("YouTube URL을 입력하세요.", key="yt_url_in")
        yt_btn = st.button("영상 분석하기", type="primary")

    if yt_btn and yt_url_input and st.session_state.yt_url != yt_url_input:
        st.session_state.yt_messages = []
        st.session_state.yt_url = yt_url_input
        with st.spinner("자막 추출 및 실시간 한글 번역 분석 중..."):
            try:
                # 한국어와 영어 자막을 모두 수집할 수 있도록 배치
                loader = YoutubeLoader.from_youtube_url(yt_url_input, add_video_info=False, language=["ko", "en"])
                docs = loader.load()
                if not docs:
                    st.error("자막을 가져올 수 없는 영상입니다.")
                    st.stop()

                full_text = docs[0].page_content
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                
                # 프롬프트에 '한글 번역' 명령 명시적 추가
                sum_prompt = ChatPromptTemplate.from_template(
                    "다음은 유튜브 영상의 자막 스크립트입니다. 영상 원문이 영어 등 외국어라면 "
                    "반드시 자연스러운 한국어로 직접 번역하여, 영상의 핵심 내용을 3~5줄의 명확한 불릿 포인트로 요약해줘.\n\n{text}"
                )
                sum_chain = sum_prompt | llm
                sum_res = sum_chain.invoke({"text": full_text[:15000]})
                st.session_state.yt_summary = sum_res.content

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = splitter.split_documents(docs)
                embeddings = OpenAIEmbeddings()
                st.session_state.yt_vector_store = InMemoryVectorStore.from_documents(splits, embeddings)
                st.success("✅ 영상 분석 및 번역 준비 완료!")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    if st.session_state.yt_summary:
        with st.expander("📌 한글 자동 번역 요약본 보기", expanded=True): st.markdown(st.session_state.yt_summary)

    st.divider()
    for msg in st.session_state.yt_messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if yt_prompt := st.chat_input("영상 내용에 대해 질문해보세요. (영문 자막도 한글로 답변합니다.)", key="yt_input"):
        if st.session_state.yt_vector_store is None:
            st.warning("👈 먼저 사이드바에서 영상 분석을 완료해주세요.")
            st.stop()

        st.session_state.yt_messages.append({"role": "user", "content": yt_prompt})
        with st.chat_message("user"): st.markdown(yt_prompt)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 문맥이 영문이어도 무조건 한국어로 번역해서 답변하도록 제어하는 시스템 프롬프트
        sys_prompt = (
            "당신은 유튜브 설명 어시스턴트입니다. 제공된 자막 문맥(Context)에 기반하여 답변하세요. "
            "만약 문서의 내용(Context)이 영어 등 외국어로 되어 있더라도, 질문자가 한눈에 알아들을 수 있도록 "
            "반드시 완벽하고 자연스러운 '한국어'로 번역 및 정리하여 친절하게 답변해야 합니다.\n\n"
            "Context:\n{context}"
        )
        prompt_template = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{input}")])

        retriever = st.session_state.yt_vector_store.as_retriever(search_kwargs={"k": 4})
        qa_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        with st.chat_message("assistant"):
            with st.spinner("자막 검색 후 번역 답변 생성 중..."):
                res = rag_chain.invoke({"input": yt_prompt})
                st.markdown(res["answer"])
        st.session_state.yt_messages.append({"role": "assistant", "content": res["answer"]})

# =========================================================================
# ⚖️ CASE 3: 전문 법률/의료 문서 조항 분석기
# =========================================================================
elif st.session_state.app_mode == "⚖️ 법률/의료 조항 분석기":
    c_head, c_home = st.columns([5, 1])
    with c_head:
        st.header("⚖️ 전문 법률/의료 문서 조항 분석기")
    with c_home:
        if st.button("🏠 포탈로 복귀", use_container_width=True, key="btn_home_az"):
            st.session_state.app_mode = "🏠 메인 포탈"
            st.rerun()
            
    st.caption("복잡한 계약서나 의료 지침의 텍스트 조항을 해체하여 성격 및 위험도를 시각화합니다.")

    if "az_data" not in st.session_state: st.session_state.az_data = None

    with st.sidebar:
        st.subheader("조항 분석 설정")
        az_type = st.selectbox("문서 유형", ["법률(계약서)", "의료(지침)"])
        sample = (
            "제1조 (목적) 本 계약은 서비스를 규정함을 목적으로 한다.\n\n"
            "제2조 (대금 지급) 을은 완료 후 7일 내 대금을 지급해야 한다. 지연 시 연 15%의 이자를 가산하며, 갑은 서비스를 즉시 중단할 수 있다.\n\n"
            "제3조 (비밀유지) 비밀정보를 누설하여서는 안 된다. 단, 법원의 명령이 있는 경우는 예외로 한다."
            if az_type == "법률(계약서)" else
            "지침 1. (환자 확인) 의약품 투여 전 환자 정보를 두 가지 이상 확인해야 한다.\n\n"
            "지침 2. (오류 보고) 오류 발생 시 30분 이내에 보고서를 제출해야 한다. 지연 시 징계 조치에 처해질 수 있다."
        )
        az_text = st.text_area("분석할 조항 입력", value=sample, height=300)
        az_btn = st.button("🔍 조항 구조화 시작", type="primary")

    if az_btn and az_text:
        with st.spinner("LLM 조항 분석 중..."):
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                sys_prompt = (
                    "당신은 전문 문서 분석가입니다. 조항을 분석하여 반드시 마크다운(```json) 없이 "
                    "아래 필드를 가진 순수 JSON 배열만 반환하세요.\n"
                    "필드: [article, type(의무/권리/제한/예외), summary, risk_level(High/Medium/Low), description]"
                )
                prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{text}")])
                chain = prompt | llm
                res = chain.invoke({"text": az_text})
                cleaned = res.content.strip().replace("```json", "").replace("```", "")
                st.session_state.az_data = json.loads(cleaned)
                st.success("🎉 분석 완료!")
            except Exception as e:
                st.error(f"오류 발생: {e}")

    if st.session_state.az_data:
        df = pd.DataFrame(st.session_state.az_data)
        m1, m2, m3 = st.columns(3)
        m1.metric("총 조항 수", f"{len(df)}개")
        m2.metric("⚠️ High Risk 조항", f"{len(df[df['risk_level']=='High'])}개")
        m3.metric("🔔 의무 조항", f"{len(df[df['type']=='의무'])}개")

        st.divider()
        st.subheader("📊 조항 대시보드 리스트")
        st.dataframe(df[['article', 'type', 'summary', 'risk_level']], use_container_width=True, hide_index=True)

        st.divider()
        for item in st.session_state.az_data:
            icon = "🔴 [고위험]" if item['risk_level'] == 'High' else "🟡 [중위험]" if item['risk_level'] == 'Medium' else "🟢"
            with st.expander(f"{icon} {item['article']} — {item['summary']}"):
                st.markdown(f"**성격:** `{item['type']}` | **위험도:** `{item['risk_level']}`")
                st.info(item['description'])

# =========================================================================
# 🗂️ 글로벌 사이드바 푸터 (모든 기능 컴포넌트가 전부 그려진 직후 최하단 고정)
# =========================================================================
with st.sidebar:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption("© 2026 Internal AI Agent Hub.")