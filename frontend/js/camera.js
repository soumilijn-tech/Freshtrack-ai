/**
 * FreshTrack AI Camera & Upload Handler
 */
export class CameraHandler {
  constructor({ onImageSelected, videoElement, modalElement }) {
    this.onImageSelected = onImageSelected;
    this.videoElement = videoElement;
    this.modalElement = modalElement;
    this.mediaStream = null;
  }

  initDropzone(dropzoneElement, fileInputElement) {
    if (!dropzoneElement || !fileInputElement) return;

    dropzoneElement.addEventListener('click', () => fileInputElement.click());

    fileInputElement.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        this.onImageSelected(file);
      }
    });

    ['dragenter', 'dragover'].forEach((name) => {
      dropzoneElement.addEventListener(name, (e) => {
        e.preventDefault();
        dropzoneElement.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((name) => {
      dropzoneElement.addEventListener(name, (e) => {
        e.preventDefault();
        dropzoneElement.classList.remove('dragover');
      });
    });

    dropzoneElement.addEventListener('drop', (e) => {
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) {
        this.onImageSelected(file);
      }
    });
  }

  async startLiveCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera access not supported by your browser.');
    }

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
        audio: false,
      });
      if (this.videoElement) {
        this.videoElement.srcObject = this.mediaStream;
        await this.videoElement.play();
      }
      if (this.modalElement) {
        this.modalElement.classList.add('active');
      }
    } catch (err) {
      throw new Error('Camera access denied or unavailable: ' + err.message);
    }
  }

  stopLiveCamera() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }
    if (this.modalElement) {
      this.modalElement.classList.remove('active');
    }
  }

  captureSnapshot() {
    if (!this.videoElement || !this.mediaStream) return null;

    const canvas = document.createElement('canvas');
    canvas.width = this.videoElement.videoWidth || 640;
    canvas.height = this.videoElement.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        const file = new File([blob], `camera_scan_${Date.now()}.jpg`, { type: 'image/jpeg' });
        this.stopLiveCamera();
        resolve(file);
      }, 'image/jpeg', 0.95);
    });
  }
}
