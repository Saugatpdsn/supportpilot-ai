# TaskNest Troubleshooting Guide

## App is slow or pages will not load
1. Refresh the page.
2. Clear your browser cache and cookies.
3. Try a private window or another browser. Supported browsers are the latest 2 versions of Chrome, Firefox, Safari, and Edge.
4. Check status.tasknest.example for ongoing incidents.

## Notifications are not arriving
Check Settings > Notifications, look in your spam folder, and add noreply@tasknest.example to your email allowlist.

## File upload fails
The maximum file size is 100 MB on Pro and Business, and 25 MB on Free. Uploads also fail when the workspace has reached its storage limit.

## Integration sync problems (Slack, Google Calendar, GitHub)
Disconnect and reconnect the integration. Syncs can take up to 15 minutes. Error SYNC-401 means the authorization expired, so re-authorize the integration. Error SYNC-429 means a rate limit was hit, so wait 10 minutes and retry.

## API errors
HTTP 401 means the API token is invalid or expired. HTTP 429 means the rate limit was exceeded: 100 requests per minute on Pro and 500 requests per minute on Business. The Free plan has no API access.

## When to contact support
If these steps do not help, contact support and include the error code, your browser and version, and a screenshot.