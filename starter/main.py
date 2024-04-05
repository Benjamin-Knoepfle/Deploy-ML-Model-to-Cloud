# Put the code for your API here.
from fastapi import FastAPI, HTTPException
from typing import Optional, Dict
from pydantic import BaseModel

class ItemPayload(BaseModel):
    item_id: Optional[int]
    item_name: str
    quantity: int

app = FastAPI()
grocery_list: Dict[int, ItemPayload] = {}


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Hello World"}

# Route to add a item
@app.post("/items/{item_name}/{quantity}")
def add_item(item_name: str, quantity: int) -> Dict[str, ItemPayload]:
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0.")
    # if item already exists, we'll just add the quantity.
    # get all item names
    items_ids: dict[str, int] = {item.item_name: item.item_id if item.item_id is not None else 0 for item in grocery_list.values()}
    if item_name in items_ids.keys():
        # get index of item_name in item_ids, which is the item_id
        item_id = items_ids[item_name]
        grocery_list[item_id].quantity += quantity
# otherwise, create a new item
    else:
        # generate an ID for the item based on the highest ID in the grocery_list
        item_id: int = max(grocery_list.keys()) + 1 if grocery_list else 0
        grocery_list[item_id] = ItemPayload(
            item_id=item_id, item_name=item_name, quantity=quantity
        )

    return {"item": grocery_list[item_id]}


# Route to list a specific item by ID
@app.get("/items/{item_id}")
def list_item(item_id: int) -> Dict[str, ItemPayload]:
    if item_id not in grocery_list:
        raise HTTPException(status_code=404, detail="Item not found.")
    return {"item": grocery_list[item_id]}


# Route to list all items
@app.get("/items")
def list_items() -> Dict[str, Dict[int, ItemPayload]]:
    return {"items": grocery_list}


# Route to delete a specific item by ID
@app.delete("/items/{item_id}")
def delete_item(item_id: int) -> Dict[str, str]:
    if item_id not in grocery_list:
        raise HTTPException(status_code=404, detail="Item not found.")
    del grocery_list[item_id]
    return {"result": "Item deleted."}


# Route to remove some quantity of a specific item by ID
@app.delete("/items/{item_id}/{quantity}")
def remove_quantity(item_id: int, quantity: int) -> Dict[str, str]:
    if item_id not in grocery_list:
        raise HTTPException(status_code=404, detail="Item not found.")
    # if quantity to be removed is higher or equal to item's quantity, delete the item
    if grocery_list[item_id].quantity <= quantity:
        del grocery_list[item_id]
        return {"result": "Item deleted."}
    else:
        grocery_list[item_id].quantity -= quantity
    return {"result": f"{quantity} items removed."}