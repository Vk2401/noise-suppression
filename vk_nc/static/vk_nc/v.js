import { createNoise3D } from "https://cdn.skypack.dev/simplex-noise";

console.log(createNoise3D);

const LINES = 4;
const LINE_COLOR = "#FFE5A3";
// const FILL_COLOR = "rgba(242, 219, 160, 0.8)"
const PRECISION = 10;
// let AMPLITUDE = 100
const FREQUENCY = 0.002;
const params = {
	amplitude: 10
};

const simplex = createNoise3D();

class Line {
	constructor(context, i = Math.random()) {
		this.context = context;
		this.canvas = context.canvas;
		this.i = i;
	}
	get stepX() {
		return this.canvas.width / PRECISION;
	}
	update(t) {
		// prevent lines crossing at the edges
		const y = (this.i - LINES * 0.5 + 0.5) * 5;
		const offset = this.canvas.height / 2 + y;
		this.points = new Array(Math.ceil(this.stepX) + 1).fill(0).map((_, i) => {
			// increase amplitude in middle
			const multiplier =
				Math.sin(
					((this.canvas.width - i * PRECISION) / this.canvas.width) * Math.PI
				) * 0.75;
			return {
				x: i * PRECISION,
				y:
					simplex(i * PRECISION * FREQUENCY, this.i, t) *
						params.amplitude *
						multiplier +
					offset
			};
		});
	}
}

class AudioWave {
	constructor(context) {
		this.context = context;
		this.canvas = context.canvas;
		this.lines = [];
		for (let i = 0; i < LINES; i++) {
			this.lines.push(new Line(context, i));
		}
	}
	get stepX() {
		return this.canvas.width / PRECISION;
	}
	draw(t) {
		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.lines.forEach((line) => line.update(t));
		this.context.globalAlpha = 0.33;
		const shape1 = [...this.lines[0].points, ...this.lines[2].points.reverse()];
		this.context.save();
		this.context.beginPath();
		shape1.forEach(({ x, y }) => {
			this.context.lineTo(x, y);
		});
		let gradient = this.context.createLinearGradient(0, 0, this.canvas.width, 0);
		gradient.addColorStop(0, "rgba(194,167,98,0)");
		gradient.addColorStop(1, "rgba(242,219,160,0.8)");
		this.context.fillStyle = gradient;
		this.context.fill();
		this.context.closePath();
		this.context.restore();
		this.context.globalAlpha = 0.5;
		const shape2 = [...this.lines[1].points, ...this.lines[3].points.reverse()];
		this.context.save();
		this.context.beginPath();
		shape2.forEach(({ x, y }) => {
			this.context.lineTo(x, y);
		});
		gradient = this.context.createLinearGradient(0, 0, 0, this.canvas.height);
		gradient.addColorStop(0.8, "rgba(194,167,98,0)");
		gradient.addColorStop(0.2, "rgba(242,219,160,0.8)");
		this.context.fillStyle = gradient;
		this.context.fill();
		this.context.closePath();
		this.context.restore();
		this.context.globalAlpha = 1;
		this.lines.forEach(({ points }) => {
			this.context.save();
			this.context.beginPath();
			points.forEach(({ x, y }, i) => {
				this.context.lineTo(x, y);
			});
			this.context.strokeStyle = LINE_COLOR;
			this.context.stroke();
			this.context.closePath();
			this.context.restore();
		});
	}
}

const root = document.querySelector("#root");
const canvas = root.querySelector("canvas");

const context = canvas.getContext("2d");

const audioWave = new AudioWave(context);

const resizeObserver = new ResizeObserver(([entry]) => {
	const { width, height } = entry.contentRect;
	canvas.width = width;
	canvas.height = height;
});

resizeObserver.observe(root);

function raf(t) {
	audioWave.draw(t * 0.001);
	requestAnimationFrame(raf);
}
requestAnimationFrame(raf);

const audio = root.querySelector("audio");
const toggle = root.querySelector("#toggle");
let play = false;

toggle.addEventListener(
	"click",
	() => {
		play = !play;

		if (play) {
			audio.play();
		} else {
			audio.pause();
		}
		toggle.classList.toggle("play", play);
		gsap.to(params, {
			duration: 1.5,
			amplitude: play ? 100 : 10,
			ease: "expo.out"
		});
	},
	false
);
