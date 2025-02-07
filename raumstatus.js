async function checkRoomStatus() {
	// Den Inhalt des Elements mit der ID "room-status" aktualisieren, um den aktuellen Status anzuzeigen
	let statusElement = document.getElementById("room-status");

	// Code hier, um den Status des Schalters abzufragen und die Variable "roomStatus" entsprechend zu setzen
	const res = await fetch('https://api.thingspeak.com/channels/2053536/fields/5.json?api_key=NSQAP62K2L94703K&results=2')

	const json = await res.json()
	// console.log('raumstatus json', json)

	const isOpen = json.feeds[0].field5 === "1"
	let roomStatus = "closed"
	if (isOpen) {
		roomStatus = "open"


		// Code hier, um den Status des Schalters abzufragen und die Variable "smokeStatus" entsprechend zu setzen
		const res = await fetch('https://api.thingspeak.com/channels/2053536/fields/7.json?api_key=NSQAP62K2L94703K&results=2')
		const json = await res.json()

		const isNoSmoke = json.feeds[0].field7 === "1"
		if (isNoSmoke) {
			roomStatus = "open - today no smoking"
		}
	}


	console.log('status', roomStatus)
	statusElement.innerHTML = roomStatus;
	if (isOpen) {
		statusElement.classList.add('open')
	} else {
		statusElement.classList.remove('open')

	}



	// recheck status after 5 seconds
	setTimeout(checkRoomStatus, 5000);

}

// check room status once document is loaded.
window.addEventListener('DOMContentLoaded', checkRoomStatus)

