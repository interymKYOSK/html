const THINGSPEAK_READ_API = 'https://api.thingspeak.com/channels/2053536/fields/5.json?api_key=NSQAP62K2L94703K&results=50';
const THINGSPEAK_SMOKE_API = 'https://api.thingspeak.com/channels/2053536/fields/7.json?api_key=NSQAP62K2L94703K&results=2';

// Function to fetch the room status history
async function fetchRoomStatus() {
	const res = await fetch(THINGSPEAK_READ_API);
	const json = await res.json();
	return json.feeds.map(feed => ({
		time: new Date(feed.created_at),
		status: feed.field5
	}));
}

// Function to check if the room status should be corrected
async function fixRoomStatus(statusElement) {
	const statusHistory = await fetchRoomStatus();

	let openStartTime = null;
	let isCurrentlyOpen = statusHistory[0].status === "1";

	// Get current time and day
	const now = new Date();
	const currentHour = now.getHours();
	const currentMinute = now.getMinutes();
	const currentDay = now.getDay(); // 0 = Sunday, 1 = Monday, ..., 6 = Saturday

	// Define opening hours
	let isInOpeningHours = false;

	if (currentDay === 0) { // Sunday
		if ((currentHour > 13 || (currentHour === 13 && currentMinute >= 12)) && currentHour < 17) {
			isInOpeningHours = true;
		}
	} else if (currentDay !== 1) { // Tuesday - Saturday (Monday is closed)
		if (currentHour >= 17 && currentHour < 22) {
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

	if (openStartTime) {
		const timeOpen = (now - openStartTime) / 1000 / 60 / 60; // Convert ms to hours
		console.log("timeOpen (hours):", timeOpen);
		if (timeOpen > 1) {
			console.log("Switch has been left open for more than 5 hours.");
			shouldBeClosed = true;
		}
	}

	// If the room should be closed, update ThingSpeak and the UI
	if (shouldBeClosed && isCurrentlyOpen) {
		console.log("Room status is CLOSED as noone should work 5 hours in a row...");
		// Overwrite the status in the UI
		statusElement.innerHTML = "closed";
		statusElement.classList.remove("open");
	} else if (!shouldBeClosed && !isCurrentlyOpen) {
		// If it should be open, update the UI to reflect that
		statusElement.innerHTML = "open";
		statusElement.classList.add("open");
	}
	setTimeout(fixRoomStatus, 15000);
}

// Function to check and display room status in the UI
async function checkRoomStatus() {
	let statusElement = document.getElementById("room-status");
	const res = await fetch(THINGSPEAK_READ_API);
	const json = await res.json();
	const isOpen = json.feeds[0].field5 === "1";
	let roomStatus = "closed";

	if (isOpen) {
		roomStatus = "open";

		// Fetch smoke status
		const resSmoke = await fetch(THINGSPEAK_SMOKE_API);
		const jsonSmoke = await resSmoke.json();
		const isNoSmoke = jsonSmoke.feeds[0].field7 === "1";
		if (isNoSmoke) {
			roomStatus = "open - today no smoking";
		}
	}

	console.log("status", roomStatus);
	statusElement.innerHTML = roomStatus;
	if (isOpen) {
		statusElement.classList.add("open");
	} else {
		statusElement.classList.remove("open");
	}

	setTimeout(checkRoomStatus, 5000);
}

// Run the checks
window.addEventListener('DOMContentLoaded', () => {
	let statusElement = document.getElementById("room-status");
	checkRoomStatus();
	fixRoomStatus(statusElement);
});
