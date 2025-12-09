const form = document.getElementById("recordForm");
const statusEl = document.getElementById("status");

const videoContainer = document.getElementById("videoContainer");
const videoEl = document.getElementById("recordedVideo");
const videoSource = document.getElementById("videoSource");

const framesContainer = document.getElementById("framesContainer");
const framesDiv = document.getElementById("frames");

const landmarksContainer = document.getElementById("landmarksContainer");
const landmarksDiv = document.getElementById("landmarks");

const resultsContainer = document.getElementById("resultsContainer");

const top3Container = document.getElementById("top3Container");


// ---------------------------
// Top-3 업데이트
// ---------------------------
function updateTop3(labels, probs) {
    top3Container.innerHTML = "";

    if (!labels || !probs) return;

    labels.forEach((label, i) => {
        const div = document.createElement("div");
        div.classList.add("top3-item");
        div.textContent = `${i + 1}. ${label} (${probs[i]}%)`;
        top3Container.appendChild(div);
    });
}


// ---------------------------
// 녹화 시작
// ---------------------------
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    statusEl.textContent = "녹화 중...";

    // 비디오 초기화
    videoSource.src = "";
    videoContainer.classList.add("hidden");

    // 예측 초기화
    predResult.innerText = "예측 결과: - (-%)";
    top3Container.innerHTML = "";
    resultsContainer.classList.add("hidden");

    await fetch("/start_recording", { method: "POST" });
});


// ---------------------------
// 파일 존재 판단
// ---------------------------
const waitForVideo = async (url, retries = 10, delay = 500) => {
    for (let i = 0; i < retries; i++) {
        const res = await fetch(url, { method: "HEAD" });
        if (res.ok) return true;
        await new Promise(r => setTimeout(r, delay));
    }
    return false;
};


// ---------------------------
// 서버 상태 모니터링
// ---------------------------
async function monitorStatus() {
    const res = await fetch("/recording_status");
    const data = await res.json();

    statusEl.textContent = data.status;
    const labels = data.predicted_labels;

    // ============================
    // 전체 프로세스 완료 → 결과 표시
    // ============================
    if (data.status.includes("전체 프로세스 완료") && labels) {
        resultsContainer.classList.remove("hidden");

        const label = labels.top1_label ?? "-";
        const prob = labels.top1_prob ?? "-";

        document.getElementById("predTitle").innerText = `예측 결과: ${top1_label} (${(top1_prob * 100).toFixed(1)}%)`;

        updateTop3(labels.top3_labels, labels.top3_probs);
    }

    // ============================
    // 녹화 영상 표시
    // ============================
    if (data.status.includes("프레임 추출 완료") && videoContainer.classList.contains("hidden")) {
        const videoUrl = `/recorded_video?time=${Date.now()}`;
        const exists = await waitForVideo(videoUrl);

        if (exists) {
            videoSource.src = videoUrl;
            videoContainer.classList.remove("hidden");
            videoEl.load();
            videoEl.play();
        }
    }
}

setInterval(monitorStatus, 1000);
