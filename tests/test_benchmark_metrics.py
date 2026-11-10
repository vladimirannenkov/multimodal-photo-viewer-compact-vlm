from benchmarks.evaluate import amounts, date_values, extract_fields, gold


def test_invoice_reference_and_value_matching():
    row = {
        "parsed_data": "{\"json\": \"{'header': {'invoice_date': '09/18/2015'}, 'summary': {'total_gross_worth': '$ 978,12'}}\"}"
    }
    date, total = gold(row)
    assert date_values(date) & date_values("Date: 09-18-2015")
    assert amounts(total) & amounts("Total $978.12")
    assert not (amounts(total) & amounts("Total 978.13"))


def test_extract_labelled_invoice_fields():
    text = "Date of issue:\n\nSeller Tax Id: 985-73-8194\n09/18/2015\n| Total | | $889,20 | $88,92 | $978,12 |"
    assert extract_fields(text) == ("09/18/2015", "978,12")
    assert extract_fields("Price 12.99") == (None, None)
