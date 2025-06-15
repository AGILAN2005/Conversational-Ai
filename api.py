from livekit.agents import llm
import enum
from typing import Annotated
import logging
from db_driver import DatabaseDriver

logger = logging.getLogger("user-data")
logger.setLevel(logging.INFO)

DB = DatabaseDriver()

class CarDetails(enum.StrEnum):  # Use StrEnum for easier formatting
    VIN = "vin"
    Make = "make"
    Model = "model"
    Year = "year"

class AssistantFnc(llm.FunctionContext):
    def __init__(self):
        super().__init__()
        self._car_details = {
            CarDetails.VIN: "",
            CarDetails.Make: "",
            CarDetails.Model: "",
            CarDetails.Year: ""
        }

    def get_car_str(self) -> str:
        return "\n".join(f"{key.value}: {value}" for key, value in self._car_details.items())

    @llm.function(description="Lookup a car by its VIN")
    async def lookup_car(
        self,
        vin: Annotated[str, llm.TypeInfo(description="The VIN of the car to look up")]
    ) -> str:
        logger.info("Looking up car - VIN: %s", vin)
        result = DB.get_car_by_vin(vin)
        if result is None:
            return "Car not found"

        self._car_details = {
            CarDetails.VIN: result.vin,
            CarDetails.Make: result.make,
            CarDetails.Model: result.model,
            CarDetails.Year: result.year
        }
        return f"The car details are:\n{self.get_car_str()}"

    @llm.function(description="Get the details of the current car")
    async def get_car_details(self) -> str:
        logger.info("Getting car details")
        return f"The car details are:\n{self.get_car_str()}"

    @llm.function(description="Create a new car")
    async def create_car(
        self,
        vin: Annotated[str, llm.TypeInfo(description="The VIN of the car")],
        make: Annotated[str, llm.TypeInfo(description="The make of the car")],
        model: Annotated[str, llm.TypeInfo(description="The model of the car")],
        year: Annotated[int, llm.TypeInfo(description="The year of the car")]
    ) -> str:
        logger.info("Creating car - VIN: %s, Make: %s, Model: %s, Year: %s", vin, make, model, year)
        result = DB.create_car(vin, make, model, year)
        if result is None:
            return "Failed to create car"

        self._car_details = {
            CarDetails.VIN: result.vin,
            CarDetails.Make: result.make,
            CarDetails.Model: result.model,
            CarDetails.Year: result.year
        }
        return "Car created!"

    def has_car(self) -> bool:
        return bool(self._car_details[CarDetails.VIN])
