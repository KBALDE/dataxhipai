

curl   -X POST -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/graphql" https://api.yelp.com/v3/graphql --data '
{
    business(id: "yelp-san-francisco") {   # the query with an argument
        name                               # a field that returns a scalar
        id                                 # a field that returns a scalar
        coordinates {                      # a field that returns an object
            latitude                       # a field that returns a scalar
            longitude                      # a field that returns a scalar
        }
    }
}'
