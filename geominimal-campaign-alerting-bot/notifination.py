import json
import requests

import config


def send_to_slack(text: str, blocks: list = None) -> None:
    result = requests.post('https://slack.com/api/chat.postMessage', {
        'token': config.slack_token,
        'channel': config.slack_channel,
        'text': text,
        'icon_emoji': ':spanchbob:',
        'username': 'incentives_analytics',
        'blocks': json.dumps(blocks) if blocks else None
    })

    if not result.ok:
        raise RuntimeError('Slack request failed')

    print('Slack response', result.text)
    response = json.loads(result.text)
    if not response['ok']:
        raise RuntimeError('Slack response is not ok')
