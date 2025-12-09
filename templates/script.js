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

function updateTop5Labels(top5_per_feature) {
    top5Container.innerHTML = "";
    if (!top5_per_feature) return;
    top5_per_feature.forEach((top5, idx) => {
        const div = document.createElement("div");
        div.textContent = `Feature ${idx + 1}: ${top5.join(", ")}`;
        top5Container.appendChild(div);
    });
}

// 녹화 버튼 클릭
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    statusEl.textContent = "녹화 중...";

    videoSource.src = "";
    videoContainer.style.display = "none";
    framesDiv.innerHTML = "";
    framesContainer.style.display = "none";
    landmarksDiv.innerHTML = "";
    landmarksContainer.style.display = "none";

    await fetch("/start_recording", { method: "POST" });
});

// 영상 준비 확인
const waitForVideo = async (url, retries = 10, delay = 500) => {
    for (let i = 0; i < retries; i++) {
        const res = await fetch(url, { method: "HEAD" });
        if (res.ok) return true;
        await new Promise(r => setTimeout(r, delay));
    }
    return false;
};

// 녹화 상태 모니터링
setInterval(async () => {
    const res = await fetch("/recording_status");
    const data = await res.json();
    statusEl.textContent = data.status;

    // 영상 자동 재생
    if (data.status.includes("프레임 추출 완료") && videoContainer.style.display === "none") {
        const videoUrl = `/recorded_video?time=${new Date().getTime()}`;
        const exists = await waitForVideo(videoUrl);
        if (exists) {
            videoSource.src = videoUrl;
            videoContainer.style.display = "block";
            videoEl.load();
            videoEl.play();
        }
    }

    // 프레임 미리보기
    if (videoContainer.style.display === "block" && data.frame_count > 0 && framesContainer.style.display === "none") {
        framesContainer.style.display = "block";
        const totalFrames = data.frame_count;
        const displayCount = 5;
        const step = Math.floor(totalFrames / displayCount) || 1;

        for (let i = 0; i < totalFrames && i < displayCount * step; i += step) {
            const img = document.createElement("img");
            img.src = `/frames/frame_${i.toString().padStart(5,'0')}.jpg?time=${new Date().getTime()}`;
            img.width = 180;
            framesDiv.appendChild(img);
        }
    }

    // 랜드마크 미리보기
    if (data.status.includes("랜드마크 추출 및 특징 생성 완료") && landmarksContainer.style.display === "none") {
        landmarksContainer.style.display = "block";
        const totalLandmarks = data.landmark_count;
        const displayCount = 5;
        const step = Math.floor(totalLandmarks / displayCount) || 1;

        for (let i = 0; i < totalLandmarks && i < displayCount * step; i += step) {
            const img = document.createElement("img");
            img.src = `/draw_landmarks/landmark_${i.toString().padStart(5,'0')}.jpg?time=${new Date().getTime()}`;
            img.width = 180;
            landmarksDiv.appendChild(img);
        }
    }
}, 1000);

// Top5 + 최종 라벨 업데이트
async function updateStatus() {
    const res = await fetch("/recording_status");
    const data = await res.json();
    statusEl.innerText = data.status;
    document.getElementById("pred_label").innerText = data.predicted_labels?.final_label || "-";
    updateTop5Labels(data.predicted_labels?.top5_per_feature);
}

setInterval(updateStatus, 1000);
