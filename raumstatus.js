const THINGSPEAK_READ_API = 'https://api.thingspeak.com/channels/2053536/fields/5.json?api_key=NSQAP62K2L94703K&minutes=400';
const THINGSPEAK_SMOKE_API = 'https://api.thingspeak.com/channels/2053536/fields/7.json?api_key=NSQAP62K2L94703K&results=2';

// Function to check and display room status in the UI
async function checkRoomStatus() {
	let statusElement = document.getElementById("room-status");
	let roomStatus = "closed";
	let openStartTime = null;
	let isInOpeningHours = false;

	const now = new Date();
	const currentHour = now.getHours();
	const currentMinute = now.getMinutes();
	const currentDay = now.getDay(); // 0 = Sunday, 1 = Monday, ..., 6 = Saturday
	const res = await fetch(THINGSPEAK_READ_API);
	const json_open = await res.json();
	const statusHistory = json_open.feeds.map(feed => ({
		time: new Date(feed.created_at), // Convert the created_at to a Date object
		status: feed.field5 // Extract the status
	}));
	// Access the latest value of field5 (status)
	const isOpen = statusHistory[statusHistory.length - 1].status === "1";

	// Define opening hours
	if (currentDay === 0) { // Sunday
		if ((currentHour > 13 || (currentHour === 13 && currentMinute >= 12)) && currentHour < 17) {
			isInOpeningHours = true;
		}
	} else if (currentDay !== 1) { // Tuesday - Saturday (Monday is closed)
		if (currentHour >= 8 && currentHour < 22) {
			isInOpeningHours = true;
		}
	}

	// Check if the switch was left open for more than 5 hours
	for (let i = 0; i < statusHistory.length; i++) {
		if (statusHistory[i].status === "1") {
			if (!openStartTime) openStartTime = statusHistory[i].time; // Mark when it first opened
		} else {
			openStartTime = null; // Reset if it was closed at any point
		}
	}

	let shouldBeClosed = !isInOpeningHours;
	const timeOpen = (now - openStartTime) / 1000 / 60 / 60; // Convert ms to hours
	if (timeOpen > 3) {
		console.log("Switch has been left open for more than 5 hours.");
		shouldBeClosed = true;
	}

	// room is closed, if outside of opening hours and 
	console.log("shouldBeClosed ", shouldBeClosed);
	console.log("isOpen ", isOpen);
	if (!isOpen || shouldBeClosed) {
		console.log("Room status is CLOSED check opening hours");
		roomStatus = "closed"
		statusElement.innerHTML = roomStatus;
		statusElement.classList.remove("open");
	} else if (isInOpeningHours && isOpen) {
		roomStatus = "open";
		// Fetch smoke status
		const resSmoke = await fetch(THINGSPEAK_SMOKE_API);
		const jsonSmoke = await resSmoke.json();
		const isNoSmoke = jsonSmoke.feeds[0].field7 === "1";
		if (isNoSmoke) {
			roomStatus = "open - today no smoking";
		}
		statusElement.innerHTML = roomStatus;
		statusElement.classList.add("open");
	}

	if (isOpen) {

	}

	setTimeout(checkRoomStatus, 5000);
}

// Run the checks
window.addEventListener('DOMContentLoaded', checkRoomStatus)
