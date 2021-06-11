## Next steps

Now that you've finished the tutorial, you might be wondering: *what's next?*

### Conduct an online A/B test
If offline scoring profile optimization resulted in a positive lift over the baseline ndcg, it's time for an online test. The control group (A) will be provided search results based on the current (baseline) scoring profile and for the test group (B) search results based on the optimized scoring profile. 
There are several ways to define which specific users are in either the control or test group and we review some methods below. The duration of this test can be determined by your average daily traffic and how much of this traffic you'll need to collect to determine whether the test is (statistically) conclusive.

#### Control/Test Method 1: Randomly split by IP addresses
Using a tool like [Azure Front door](https://docs.microsoft.com/en-us/azure/frontdoor/front-door-overview#:~:text=Azure%20Front%20Door%20is%20a,and%20widely%20scalable%20web%20applications.&text=Front%20Door%20provides%20a%20range,needs%20and%20automatic%20failover%20scenarios.), a coin is (programmatically) tossed each time a new IP address is encountered during the experiment. If it's heads, that IP address becomes part of the control group; otherwise, if it's tails then it is part of the test group. One adnvantage of this method is that it is scalable and simple to implement. One potential disadvantage is that a customer may not uniquely map to an IP address, e.g. because they are using different devices or browser's in incognito mode.


#### Control/Test Method 2: Random asssignment by geo-location

### Deploy the model
If your first A/B experiment shows a positive lift in favor of the test group (B), you may want to deploy the offline optimization so that it runs at a fixed cadence (e.g. once a month) or in response to newly created articles in your search index.

### Best practices for model deployment

