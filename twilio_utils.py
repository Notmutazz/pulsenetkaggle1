from twilio.rest import Client
import config

def _generate_simulated_mistral_advice(alert_data):
    """
    This is our SIMULATED Mistral function. It generates the detailed
    safety advice that will be sent in the SMS.
    """
    disaster_type = alert_data.get("disaster_type")
    location = alert_data.get("location", "your area")

    if disaster_type == 'flood':
        rainfall = alert_data.get('rainfall', 'high levels')
        return (
            f"PulseNet ALERT for {location}:\n\n"
            f"A severe FLOOD warning is in effect with rainfall at {rainfall:.1f}mm/hr. "
            "Move to higher ground and avoid all non-essential travel. Please secure your important documents."
        )
    elif disaster_type == 'heatwave':
        temperature = alert_data.get('temperature', 'dangerous levels')
        return (
            f"PulseNet ALERT for {location}:\n\n"
            f"An extreme HEATWAVE warning is in effect with temperatures near {temperature:.1f}°C. "
            "Stay indoors during peak hours, drink plenty of water, and check on elderly neighbors."
        )
    else:
        # This function won't be called for 'normal', but it's good practice
        return "PulseNet: Conditions are currently normal in your area."

def send_alert(disaster_type, weather_data):
    """
    Sends an SMS alert using Twilio with AI-simulated advice.
    """
    # Combine all the information needed for the message
    alert_info = weather_data.copy()
    alert_info["disaster_type"] = disaster_type
    alert_info["location"] = "Rajanukunte" # We can hardcode a location for the demo

    # 1. Generate the rich, AI-simulated message using our new function
    message_body = _generate_simulated_mistral_advice(alert_info)

    # 2. Send this new message using Twilio
    try:
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            to=config.MY_PHONE_NUMBER,
            from_=config.TWILIO_PHONE_NUMBER,
            body=message_body
        )
        print(f"✅ AI-powered SMS for {disaster_type} sent!")
    except Exception as e:
        print(f"❌ Error sending SMS: {e}")