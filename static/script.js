const form = document.getElementById("recordForm");
const statusEl = document.getElementById("status");

const videoContainer = document.getElementById("videoContainer");
const videoEl = document.getElementById("recordedVideo");
const videoSource = document.getElementById("videoSource");

const framesContainer = document.getElementById("framesContainer");
const framesDiv = document.getElementById("frames");

const landmarksContainer = document.getElementById("landmarksContainer");
const landmarksDiv = document.getElementById("landmarks");

const top5Container = document.getElementById("top5Container");

const predTop1Label = document.getElementById("pred_top1_label");
const predTop1Prob = document.getElementById("pred_top1_prob");


// ---------------------------
// Top5 업데이트
// ---------------------------
function updateTop5(labels, probs) {
    top5Container.innerHTML = "";

    if (!labels || !probs) return;

    labels.forEach((label, i) => {
        const div = document.createElement("div");
        div.textContent = `${i + 1}. ${label} (${probs[i]})`;
        top5Container.appendChild(div);
    });
}


// ---------------------------
// 녹화 시작 버튼
// ---------------------------
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    statusEl.textContent = "녹화 중...";

    // 영상 초기화
    videoSource.src = "";
    videoContainer.classList.add("hidden");

    // 프레임 초기화
    framesDiv.innerHTML = "";
    framesContainer.classList.add("hidden");

    // 랜드마크 초기화
    landmarksDiv.innerHTML = "";
    landmarksContainer.classList.add("hidden");

    // 예측 라벨 초기화
    predTop1Label.innerText = "-";
    predTop1Prob.innerText = "-";
    top5Container.innerHTML = "";

    await fetch("/start_recording", { method: "POST" });
});


// ---------------------------
// 비디오 파일 존재 확인
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
    // top1 + top5 업데이트
    // ============================
    if (labels) {
        predTop1Label.innerText = labels.top1_label ?? "-";
        predTop1Prob.innerText = labels.top1_prob ?? "-";

        updateTop5(labels.top5_labels, labels.top5_probs);
    }

    // ============================
    // 영상 표시
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

    // ============================
    // 프레임 표시
    // ============================
    if (data.frame_count > 0 && videoContainer.classList.contains("hidden") === false && framesContainer.classList.contains("hidden")) {
        framesContainer.classList.remove("hidden");
        framesDiv.innerHTML = "";

        const total = data.frame_count;
        const displayCount = 5;
        const step = Math.floor(total / displayCount) || 1;

        for (let i = 0; i < total && i < displayCount * step; i += step) {
            const img = document.createElement("img");
            img.src = `/frames/frame_${i.toString().padStart(5, "0")}.jpg?time=${Date.now()}`;
            framesDiv.appendChild(img);
        }
    }

    // ============================
    // 랜드마크 표시
    // ============================
    if (data.status.includes("추론 중") && data.landmark_count > 0 && landmarksContainer.classList.contains("hidden")) {
        landmarksContainer.classList.remove("hidden");
        landmarksDiv.innerHTML = "";

        const total = data.landmark_count;
        const displayCount = 5;
        const step = Math.floor(total / displayCount) || 1;

        for (let i = 0; i < total && i < displayCount * step; i += step) {
            const img = document.createElement("img");
            img.src = `/draw_landmarks/landmark_${i.toString().padStart(5, "0")}.jpg?time=${Date.now()}`;
            landmarksDiv.appendChild(img);
        }
    }
}

setInterval(monitorStatus, 1000);
