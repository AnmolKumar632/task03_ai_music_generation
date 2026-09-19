import unittest
import json
from app import app


class TestWebApp(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'NeuralHarmony', response.data)

    def test_api_status(self):
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('device', data)
        self.assertIn('vocab_size', data)

    def test_api_history(self):
        response = self.client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('history', data)
        self.assertIsInstance(data['history'], list)

    def test_api_generate(self):
        payload = {
            "num_notes": 15,
            "temperature": 0.8,
            "seed_token": "60",
            "step_duration": 0.4
        }
        response = self.client.post(
            '/api/generate',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get('success'))
        self.assertIn('notes', data)
        self.assertIn('download_url', data)
        self.assertGreater(len(data['notes']), 0)


    def test_api_generate_with_custom_seed_and_instrument(self):
        payload = {
            "num_notes": 20,
            "temperature": 1.0,
            "seed_notes": ["60", "64", "67"],
            "instrument": "synth_lead",
            "step_duration": 0.25
        }
        response = self.client.post(
            '/api/generate',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data.get('success'))
        self.assertGreaterEqual(len(data['notes']), 20)


if __name__ == '__main__':
    unittest.main()
