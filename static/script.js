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
const predLabelEl = document.getElementById("pred_label"); // 예측 라벨 엘리먼트 추가
const predProbEl = document.getElementById("pred_prob"); // 확률 표시 엘리먼트 추가


function updateTop5Labels(top5_per_feature, top5_probs_per_feature) {
    top5Container.innerHTML = "";
    if (!top5_per_feature || !top5_probs_per_feature) return;

    top5_per_feature.forEach((top5, idx) => {
        const probs = top5_probs_per_feature[idx];
        const div = document.createElement("div");
        div.innerHTML = `Feature ${idx + 1}: ` + top5.map((label, i) => `${label} (${probs[i]})`).join(", ");
        top5Container.appendChild(div);
    });
}


// 녹화 버튼 클릭
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    statusEl.textContent = "녹화 중...";

    // 영상 초기화 및 숨기기
    videoSource.src = "";
    videoContainer.classList.add("hidden");

    // 프레임 초기화 및 숨기기
    framesDiv.innerHTML = "";
    framesContainer.classList.add("hidden");

    // 랜드마크 초기화 및 숨기기
    landmarksDiv.innerHTML = "";
    landmarksContainer.classList.add("hidden");

    // 예측 라벨 초기화
    predLabelEl.innerText = "-";
    top5Container.innerHTML = "";


    // 서버에 녹화 시작 요청
    await fetch("/start_recording", { method: "POST" });
});

// 영상 준비 확인 함수
const waitForVideo = async (url, retries = 10, delay = 500) => {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await fetch(url, { method: "HEAD" });
            if (res.ok) return true;
        } catch (error) {
            // 네트워크 오류 등 발생 시 재시도
            console.error("Video check failed, retrying:", error);
        }
        await new Promise(r => setTimeout(r, delay));
    }
    return false;
};

// 녹화 상태 모니터링 및 업데이트
async function monitorStatus() {
    const res = await fetch("/recording_status");
    const data = await res.json();
    statusEl.textContent = data.status;

    // 최종 라벨 및 확률
    predLabelEl.innerText = data.predicted_labels?.final_label || "-";
    predProbEl.innerText = data.predicted_labels?.final_prob != null ? data.predicted_labels.final_prob : "-";

    // feature별 top5 + 확률
    updateTop5Labels(data.predicted_labels?.top5_per_feature, data.predicted_labels?.top5_probs_per_feature);

    // 영상 자동 재생
    if (data.status.includes("프레임 추출 완료") && videoContainer.classList.contains("hidden")) {
        const videoUrl = `/recorded_video?time=${new Date().getTime()}`;
        const exists = await waitForVideo(videoUrl);
        if (exists) {
            videoSource.src = videoUrl;
            videoContainer.classList.remove("hidden");
            videoEl.load();
            videoEl.play();
        }
    }

    // 프레임 미리보기 → 영상이 표시될 때만 동작
    if (!videoContainer.classList.contains("hidden") && data.frame_count > 0 && framesContainer.classList.contains("hidden")) {
        framesContainer.classList.remove("hidden");
        
        // 이전에 추가된 이미지가 있다면 제거 (중복 방지)
        framesDiv.innerHTML = ""; 

        const totalFrames = data.frame_count;
        const displayCount = 5;
        const step = Math.floor(totalFrames / displayCount) || 1;

        for (let i = 0; i < totalFrames && i < displayCount * step; i += step) {
            const img = document.createElement("img");
            // 캐싱 방지를 위해 timestamp 추가
            img.src = `/frames/frame_${i.toString().padStart(5,'0')}.jpg?time=${new Date().getTime()}`;
            // img.width는 CSS로 관리됨
            framesDiv.appendChild(img);
        }
    }

    // 랜드마크 미리보기
    if (data.status.includes("랜드마크 추출 및 특징 생성 완료") && landmarksContainer.classList.contains("hidden")) {
        landmarksContainer.classList.remove("hidden");

        // 이전에 추가된 이미지가 있다면 제거 (중복 방지)
        landmarksDiv.innerHTML = "";

        const totalLandmarks = data.landmark_count;
        const displayCount = 5;
        const step = Math.floor(totalLandmarks / displayCount) || 1;

        for (let i = 0; i < totalLandmarks && i < displayCount * step; i += step) {
            const img = document.createElement("img");
            // 캐싱 방지를 위해 timestamp 추가
            img.src = `/draw_landmarks/landmark_${i.toString().padStart(5,'0')}.jpg?time=${new Date().getTime()}`;
            // img.width는 CSS로 관리됨
            landmarksDiv.appendChild(img);
        }
    }
}

// 1초마다 상태 업데이트 함수 실행
setInterval(monitorStatus, 1000);