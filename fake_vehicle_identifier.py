class FakeVehicleIdentifier:
    """
    Temporary vehicle identifier for MVP demos while a real identifier
    provider is unavailable.

    It preserves the same interface as VehicleClassifier:
        classify(crop, track_id=None) -> ("YYYY Make Model", confidence)
    """

    VEHICLES = [
        "2020 Toyota Camry",
        "2020 Honda Civic",
        "2020 Ford F-150",
        "2020 Chevrolet Silverado 1500",
        "2020 Honda Odyssey",
        "2020 Hyundai Sonata",
        "2020 Nissan Altima",
        "2020 Toyota RAV4",
    ]

    def __init__(self):
        self.cache = {}
        self.next_index = 0

    def classify(self, crop, track_id=None):
        if track_id is not None and track_id in self.cache:
            return self.cache[track_id]

        label = self.VEHICLES[self.next_index % len(self.VEHICLES)]
        self.next_index += 1
        result = (label, 0.99)

        if track_id is not None:
            self.cache[track_id] = result

        return result
