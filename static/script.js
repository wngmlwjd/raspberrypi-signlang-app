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

const predLabelProbEl = document.getElementById("pred_label_prob");
const predProbEl = document.getElementById("pred_prob");
const predLabelVoteEl = document.getElementById("pred_label_vote");

// ------------------------------------------------------
//  Top5 UI 업데이트
// ------------------------------------------------------
function updateTop5Labels(top5_per_feature, top5_probs_per_feature) {
    top5Container.innerHTML = "";

    if (!top5_per_feature || !top5_probs_per_feature) return;

    top5_per_feature.forEach((labels, idx) => {
        const probs = top5_probs_per_feature[idx];
        const wrap = document.createElement("div");
        wrap.className = "top5-item";

        let html = `<b>Feature ${idx + 1}</b><br>`;
        html += labels.map((label, i) => {
            return `${label} (${probs[i]})`;
        }).join("<br>");

        wrap.innerHTML = html;
        top5Container.appendChild(wrap);
    });
}

// ------------------------------------------------------
//  녹화 시작 버튼
// ------------------------------------------------------
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    statusEl.textContent = "녹화 중...";

    videoContainer.classList.add("hidden");
    videoSource.src = "";

    framesContainer.classList.add("hidden");
    framesDiv.innerHTML = "";

    landmarksContainer.classList.add("hidden");
    landmarksDiv.innerHTML = "";

    predLabelProbEl.innerText = "-";
    predProbEl.innerText = "-";
    predLabelVoteEl.innerText = "-";

    top5Container.innerHTML = "";

    await fetch("/start_recording", { method: "POST" });
});

// ------------------------------------------------------
//  녹화 상태 모니터링
// ------------------------------------------------------
async function monitorStatus() {
    const res = await fetch("/recording_status");
    const data = await res.json();

    statusEl.textContent = data.status;
    const labels = data.predicted_labels;

    // --- 최종 라벨 ---
    predLabelProbEl.innerText = labels?.final_label_prob || "-";
    predProbEl.innerText = labels?.final_prob ?? "-";
    predLabelVoteEl.innerText = labels?.final_label_vote || "-";

    // --- Top5 ---
    updateTop5Labels(labels?.top5_per_feature, labels?.top5_probs_per_feature);

    // --- 영상 준비되면 재생 ---
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

    // --- 프레임 미리보기 ---
    if (!videoContainer.classList.contains("hidden") && data.frame_count > 0 && framesContainer.classList.contains("hidden")) {
        framesContainer.classList.remove("hidden");
        framesDiv.innerHTML = "";

        const total = data.frame_count;
        const count = 5;
        const step = Math.floor(total / count) || 1;

        for (let i = 0; i < total && i < count * step; i += step) {
            const img = document.createElement("img");
            img.src = `/frames/frame_${i.toString().padStart(5, '0')}.jpg?time=${Date.now()}`;
            framesDiv.appendChild(img);
        }
    }

    // --- 랜드마크 미리보기 ---
    if (data.status.includes("랜드마크 추출 및 특징 생성 완료") && landmarksContainer.classList.contains("hidden")) {
        landmarksContainer.classList.remove("hidden");
        landmarksDiv.innerHTML = "";

        const total = data.landmark_count;
        const count = 5;
        const step = Math.floor(total / count) || 1;

        for (let i = 0; i < total && i < count * step; i += step) {
            const img = document.createElement("img");
            img.src = `/draw_landmarks/landmark_${i.toString().padStart(5,'0')}.jpg?time=${Date.now()}`;
            landmarksDiv.appendChild(img);
        }
    }
}

// 영상 존재 체크
async function waitForVideo(url, retries = 10, delay = 500) {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await fetch(url, { method: "HEAD" });
            if (res.ok) return true;
        } catch (e) {}
        await new Promise(r => setTimeout(r, delay));
    }
    return false;
}

setInterval(monitorStatus, 1000);
