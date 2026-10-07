# Meeting Record

## Summary
The team confirmed that services have been moved to Kubernetes and the PostgreSQL migration is complete. They plan to write Terraform scripts by the 14th and need someone to look at gRPC timeouts. Load testing showed stability up to 40,000 requests per hour but not above 50,000, prompting a discussion about switching to Redis for caching. The release date was agreed to be the 20th, and documentation still needs to be updated.

## Minutes
### Backend updates
- Moved services to Kubernetes last month
- PostgreSQL migration finished
### Terraform scripts
- Write Terraform scripts by the 14th
### gRPC timeouts
- Look at the gRPC timeouts
### Load testing
- Handled about 40,000 requests per hour; not stable above 50,000
### Caching
- Consider switching to Redis for caching
### Release schedule
- Release agreed to go out on the 20th, not earlier
### Documentation
- Someone needs to update the documentation

## Key Decisions
- Release will go out on the 20th, not earlier

## Action Items
- Write Terraform scripts | Owner: Unspecified | Deadline: 14th
- Look at the gRPC timeouts | Owner: Unspecified | Deadline: Unspecified
- Update the documentation | Owner: Unspecified | Deadline: Unspecified